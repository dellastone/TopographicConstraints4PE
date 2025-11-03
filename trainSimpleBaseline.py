# trainSimpleBaseline.py
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import MPIIDataset, load_mpii_annotations
from utils import plot_image_with_heatmaps, compute_pckh_from_heatmaps, EarlyStopping
from model import SimpleBaseline
import topo_analysis as topo_analysis

from tqdm import tqdm
import wandb as wandb
import time
import csv


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train for one epoch on 'loader'.
    Returns epoch-averaged loss and PCKh.
    """
    model.train()
    running_loss = 0.0
    running_pckh = 0.0

    loop = tqdm(loader, desc="Training", leave=False)
    for images, heatmaps in loop:
        images = images.to(device)
        heatmaps = heatmaps.to(device)

        # Forward + loss
        preds = model(images)
        loss = criterion(preds, heatmaps)

        # Metric (PCKh tuple; index 1 is mean over joints/batch)
        pckh = compute_pckh_from_heatmaps(preds, heatmaps)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate sums for epoch averaging
        bs = images.size(0)
        running_loss += loss.item() * bs
        running_pckh += pckh[1] * bs

        # Live progress (normalized by processed items)
        denom = (loop.n + 1) * loader.batch_size
        loop.set_postfix(
            train_loss=f"{running_loss / denom:.4f}",
            train_pckh=f"{running_pckh / denom:.4f}"
        )

    n = len(loader.dataset)
    return running_loss / n, running_pckh / n


def validate(model, loader, criterion, device, wandblog: bool = False, save_plots: bool = False, save_dir: str = None):
    """
    Validation loop (no grad).
    Also renders heatmaps for first image of the last batch for a quick visual check.
    """
    model.eval()
    running_loss = 0.0
    running_pckh = 0.0

    loop = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for images, heatmaps in loop:
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            preds = model(images)
            loss = criterion(preds, heatmaps)
            pckh = compute_pckh_from_heatmaps(preds, heatmaps)

            bs = images.size(0)
            running_loss += loss.item() * bs
            running_pckh += pckh[1] * bs

            denom = (loop.n + 1) * loader.batch_size
            loop.set_postfix(
                val_loss=f"{running_loss / denom:.4f}",
                val_pckh=f"{running_pckh / denom:.4f}"
            )

        # Qualitative snapshot (first image of last batch)
        plot_image_with_heatmaps(
            images[0], preds[0], radius=4, color=(0, 255, 0),
            wandblog=wandblog, save_plots=save_plots, save_dir=save_dir
        )


    n = len(loader.dataset)
    return running_loss / n, running_pckh / n


def add_argument_parser():
    """CLI for training/eval of SimpleBaseline."""
    parser = argparse.ArgumentParser(description="Train and validate TopoBaseline model.")
    parser.add_argument('--device', default='cuda', help='Device to use (e.g., cuda:0, cpu)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--val-split', type=float, default=0.1, help='Fraction of data for validation')
    parser.add_argument('--save-dir', default='checkpoints', help='Directory to save best model')

    # Logging and plotting toggles
    parser.add_argument('--wandb', dest='wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false', help='Disable Weights & Biases logging')
    parser.set_defaults(wandb=False)

    parser.add_argument('--save-plts', dest='save_plts', action='store_true', help='Enable saving plots')
    parser.add_argument('--no-save-plts', dest='save_plts', action='store_false', help='Disable saving plots')
    parser.set_defaults(save_plts=False)

    parser.add_argument('--num-keypoints', type=int, default=16, help='Number of keypoints (MPII uses 16)')

    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='Use pretrained backbone')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', help='Do not use pretrained backbone')
    parser.set_defaults(pretrained=True)

    parser.add_argument('--arch', choices=['resnet18','resnet34','resnet50','resnet101','resnet152'],
                        default='resnet18', help='Backbone architecture')

    parser.add_argument('--mat-path', default="./mpii/mpii_human_pose_v1_u12_1.mat",
                        help='Path to MPII .mat annotations file')
    parser.add_argument('--img-dir', default="./mpii/images",
                        help='Directory containing MPII images')
    return parser


def main():
    parser = add_argument_parser()
    args = parser.parse_args()

    # Local aliases for readability
    WANDBLOG      = args.wandb
    NUM_KEYPOINTS = args.num_keypoints
    PRETRAINED    = args.pretrained
    ARCHITECTURE  = args.arch
    EPOCHS        = args.epochs
    MAT_PATH      = args.mat_path
    IMG_DIR       = args.img_dir
    SAVE_PLOTS    = args.save_plts

    print(args)
    print(f"Using architecture: {ARCHITECTURE}, pretrained={PRETRAINED}")

    # Respect user device but fall back to CPU if CUDA not available
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ---- Data ----
    annotations = load_mpii_annotations(MAT_PATH)
    random.shuffle(annotations)  # simple augmentation through order

    # Train/val split (by indices) then dataset
    val_size = int(len(annotations) * args.val_split)
    train_size = len(annotations) - val_size
    full_dataset = MPIIDataset(annotations, IMG_DIR)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # ---- Run naming / logging ----
    model_name = "SimpleBaseline"
    run_name = f"{ARCHITECTURE}_{model_name}_time_{int(time.time())}"

    metrics_dir = None
    if WANDBLOG:
        wandb.init(project="ABNS", config=args, name=run_name)

    # Optional metrics/plots folder
    if SAVE_PLOTS:
        headers = ["epoch", "train_loss", "val_loss", "train_pckh", "val_pckh", "moransI_strength", "contiguity"]
        os.makedirs("plots", exist_ok=True)
        metrics_dir = os.path.join("plots", run_name)
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_csv = os.path.join(metrics_dir, "metrics.csv")
        if not os.path.exists(metrics_csv):
            with open(metrics_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    # ---- Model / loss / optim ----
    mse_loss_fn = nn.MSELoss()

    def Criterion(preds, targets):
        """Simple MSE between predicted and target heatmaps."""
        return mse_loss_fn(preds, targets)

    print("Using BASELINE MODEL")
    model = SimpleBaseline(NUM_KEYPOINTS, ARCHITECTURE, PRETRAINED).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Stop when val loss doesn't improve for 'patience' epochs
    early_stopping = EarlyStopping(patience=7, verbose=True, name=run_name)

    # Best checkpoint directory
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')

    # ---- Training loop ----
    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")

        train_loss, train_pckh = train_one_epoch(model, train_loader, Criterion, optimizer, device)

        model.eval()
        val_loss, val_pckh = validate(
            model, val_loader, Criterion, device,
            wandblog=WANDBLOG, save_plots=SAVE_PLOTS,
            save_dir=os.path.join("plots", run_name) if SAVE_PLOTS else None
        )

        # Early stopping step
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        # Log scalars
        wandbLossLog = {"train_loss": train_loss, "val_loss": val_loss,
                        "train_pckh": train_pckh, "val_pckh": val_pckh}
        if WANDBLOG:
            wandb.log(wandbLossLog, step=epoch)

        print(f"Epoch {epoch:03d} ▶ Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | "
              f"Train PCKh = {train_pckh:.4f} | Val PCKh = {val_pckh:.4f}")

        # Optional diagnostics + CSV
        if WANDBLOG or SAVE_PLOTS:
            # Parcellation map
            pref_b = topo_analysis.plot_keypoint_parcellation_from_head(
                model, epoch=epoch, wandblog=WANDBLOG, save_plots=SAVE_PLOTS, save_dir=metrics_dir
            )

            # Co-activation vs. distance curve (non-topographic baseline)
            topo_analysis.plot_coactivation_distance_curve(
                model, val_loader, device, topo=False, epoch=epoch,
                wandblog=WANDBLOG, save_plots=SAVE_PLOTS, save_dir=metrics_dir
            )

            # Neighbor correlation of activations
            topo_analysis.plot_activation_neighbor_corr(
                model.deconv, model, val_loader, device, topo=False,
                title_prefix="Classic activation neighbor corr", max_batches=64, epoch=epoch,
                wandblog=WANDBLOG, save_plots=SAVE_PLOTS, save_dir=metrics_dir
            )

            # Scalar spatial metrics
            I_b, _ = topo_analysis.morans_I_strength(model, topo=False, grid_hw=(16, 16))
            P_c = topo_analysis.parcellation_contiguity(pref_b)

            if WANDBLOG:
                wandb.log({"metrics/moransI_strength/classic": I_b}, step=epoch)
                wandb.log({"metrics/contiguity/classic": P_c}, step=epoch)

            if SAVE_PLOTS:
                with open(os.path.join(metrics_dir, "metrics.csv"), "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, train_loss, val_loss, train_pckh, val_pckh, I_b, P_c])

        # Save best checkpoint by val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, run_name + ".pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model to {save_path}")

    if WANDBLOG:
        wandb.finish()


if __name__ == '__main__':
    main()
