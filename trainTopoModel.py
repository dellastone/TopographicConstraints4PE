import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import MPIIDataset, load_mpii_annotations
from utils import plot_image_with_heatmaps, compute_pckh_from_heatmaps, EarlyStopping
from model import TopoModel
import topo_analysis as topo_analysis

from tqdm import tqdm
import wandb as wandb
import time
import math
import csv


def train_one_epoch(model, loader, criterion, optimizer, device, epoch=None, max_epochs=None):
    """
    Train for one epoch.
    Returns epoch-averaged totals for: total loss, MSE, topographic (ws) loss, and PCKh.
    """
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_ws = 0.0
    running_pckh = 0.0

    loop = tqdm(loader, desc="Training", leave=False)
    for images, heatmaps in loop:
        images = images.to(device)
        heatmaps = heatmaps.to(device)

        # Forward pass
        preds = model(images)

        # Combined criterion returns (total, mse, ws, curr_lambda)
        loss, mse_loss, ws_loss, spatial_lambda = criterion(preds, heatmaps, epoch, max_epochs)

        # Metric on predictions (PCKh@0.5 returned as tuple -> take mean at index 1)
        pckh = compute_pckh_from_heatmaps(preds, heatmaps)

        # Standard optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate sums (we'll divide by dataset size later)
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_mse += mse_loss.item() * batch_size
        running_ws += ws_loss.item() * batch_size
        running_pckh += pckh[1] * batch_size

        # Live progress display (normalized by seen items)
        denom = (loop.n + 1) * loader.batch_size
        loop.set_postfix(
            train_loss=f"{running_loss / denom:.4f}",
            mse=f"{running_mse / denom:.4f}",
            ws=f"{running_ws / denom:.4f}",
            spatial_lambda=f"{spatial_lambda:.4f}",
            train_pckh=f"{running_pckh / denom:.4f}",
        )

    # Convert sums to dataset-wide averages
    n = len(loader.dataset)
    return (
        running_loss / n,
        running_mse / n,
        running_ws / n,
        running_pckh / n,
    )


def validate(model, loader, criterion, device, epoch=None, max_epochs=None, wandblog: bool = False, save_plots: bool = False, save_dir: str = None):
    """
    Evaluate on validation set (no grad).
    Also plots one sample (first image of last batch) for quick visual sanity check.
    """
    model.eval()
    running_loss = 0.0
    running_mse = 0.0
    running_spatial = 0.0
    running_pckh = 0.0

    loop = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for images, heatmaps in loop:
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            preds = model(images)
            loss, mse_loss, ws_loss, _ = criterion(preds, heatmaps, epoch, max_epochs)
            pckh = compute_pckh_from_heatmaps(preds, heatmaps)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_mse += mse_loss.item() * batch_size
            running_spatial += ws_loss.item() * batch_size
            running_pckh += pckh[1] * batch_size

            denom = (loop.n + 1) * loader.batch_size
            loop.set_postfix(
                val_loss=f"{running_loss / denom:.4f}",
                val_mse=f"{running_mse / denom:.4f}",
                val_spatial=f"{running_spatial / denom:.4f}",
                val_pckh=f"{running_pckh / denom:.4f}",
            )

    # Quick qualitative check: show heatmaps for the first image in the last batch
    plot_image_with_heatmaps(images[0], preds[0], radius=4, color=(0, 255, 0),
                             wandblog=wandblog, save_plots=save_plots, save_dir=save_dir)

    n = len(loader.dataset)
    return (
        running_loss / n,
        running_mse / n,
        running_spatial / n,
        running_pckh / n,
    )


def add_argument_parser():
    """CLI arguments for training and evaluation."""
    parser = argparse.ArgumentParser(description="Train and validate TopoBaseline model.")

    parser.add_argument('--device', default='cuda', help='Device to use (e.g. cuda:0, cpu)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--val-split', type=float, default=0.1, help='Fraction of data for validation')
    parser.add_argument('--save-dir', default='checkpoints', help='Directory to save best model')
    parser.add_argument('--spatial-lambda', type=float, default=0.5, help="Weight λ for spatial/topographic loss")
    parser.add_argument('--topo-grid', type=int, nargs=2, default=[16, 16],
                        help="Gh Gw grid for topographic layer (Gh*Gw ideally 256)")

    # Logging + plotting flags
    parser.add_argument('--wandb', dest='wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false', help='Disable Weights & Biases logging')
    parser.set_defaults(wandb=True)

    parser.add_argument('--save-plts', dest='save_plts', action='store_true', help='Enable saving plots')
    parser.add_argument('--no-save-plts', dest='save_plts', action='store_false', help='Disable saving plots')
    parser.set_defaults(save_plts=False)

    parser.add_argument('--num-keypoints', type=int, default=16, help='Number of keypoints (MPII=16)')

    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='Use pretrained backbone')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', help='Do not use pretrained backbone')
    parser.set_defaults(pretrained=True)

    parser.add_argument('--arch', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        default='resnet18', help='Backbone architecture')

    parser.add_argument('--mat-path', default="./mpii/mpii_human_pose_v1_u12_1.mat",
                        help='Path to MPII .mat annotations file')
    parser.add_argument('--img-dir', default="./mpii/images",
                        help='Directory containing MPII images')
    return parser


def main():
    parser = add_argument_parser()
    args = parser.parse_args()

    # Shorthand locals (easier to read below)
    WANDBLOG = args.wandb
    SAVE_PLOTS = args.save_plts
    NUM_KEYPOINTS = args.num_keypoints
    PRETRAINED = args.pretrained
    ARCHITECTURE = args.arch
    EPOCHS = args.epochs
    MAT_PATH = args.mat_path
    IMG_DIR = args.img_dir
    SPATIAL_LAMBDA = args.spatial_lambda
    TOPO_GRID = args.topo_grid

    print(args)
    print(f"Using architecture: {ARCHITECTURE}, pretrained={PRETRAINED}")

    # Respect requested device but fall back to CPU if CUDA unavailable
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ---- Data ----
    annotations = load_mpii_annotations(MAT_PATH)
    random.shuffle(annotations)

    # Split indices once, then wrap with dataset
    val_size = int(len(annotations) * args.val_split)
    train_size = len(annotations) - val_size

    full_dataset = MPIIDataset(annotations, IMG_DIR)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ---- Run naming / logging setup ----
    model_name = "TopoModelLambdaScheduled"
    run_name = f"{ARCHITECTURE}_{model_name}_time_{int(time.time())}"

    metrics_dir = None
    if WANDBLOG:
        wandb.init(project="ABNS", config=args, name=run_name)

    # Optional on-disk plot/metrics folder
    if SAVE_PLOTS:
        headers = ["train_loss", "val_loss", "train_pckh", "val_pckh",
                   "train_mse", "val_mse", "train_ws", "val_ws",
                   "moransI_strength", "contiguity"]
        os.makedirs("plots", exist_ok=True)
        metrics_dir = os.path.join("plots", run_name)
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_csv = os.path.join(metrics_dir, "metrics.csv")
        if not os.path.exists(metrics_csv):
            with open(metrics_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    # ---- Losses / model / optim ----
    mse_loss_fn = nn.MSELoss()

    def Criterion(preds, targets, epoch=None, max_epochs=None, min_scale=0.1):
        """
        Composite objective = MSE + (scheduled λ)*TopoReg.
        Uses cosine schedule to anneal λ from 1.0 -> min_scale over training.
        """
        # Cosine schedule on spatial λ (start high, gradually relax)
        scale = 1.0
        if (epoch is not None) and (max_epochs is not None) and max_epochs > 1:
            cosv = 0.5 * (1 + math.cos(math.pi * epoch / (max_epochs - 1)))
            scale = min_scale + (1.0 - min_scale) * cosv

        mse_loss = mse_loss_fn(preds, targets)
        ws_loss = model.topo_reg_loss(SPATIAL_LAMBDA * scale)  # model provides topo regularizer
        loss = mse_loss + ws_loss
        return loss, mse_loss, ws_loss, SPATIAL_LAMBDA * scale

    print("Using TOPO MODEL")
    model = TopoModel(NUM_KEYPOINTS, ARCHITECTURE, PRETRAINED, TOPO_GRID[0], TOPO_GRID[1]).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Early stopping monitors val loss and stops after 'patience' epochs without improvement
    early_stopping = EarlyStopping(patience=7, verbose=True, name=run_name)

    # Best checkpoint folder
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')

    # ---- Train loop ----
    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")

        train_loss, train_mse, train_ws, train_pckh = train_one_epoch(
            model, train_loader, Criterion, optimizer, device, epoch, EPOCHS
        )

        model.eval()
        val_loss, val_mse, val_ws, val_pckh = validate(
            model, val_loader, Criterion, device, epoch, EPOCHS,
            wandblog=WANDBLOG, save_plots=SAVE_PLOTS, save_dir=os.path.join("plots", run_name) if SAVE_PLOTS else None
        )

        # Early stopping update
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        # Logging (W&B)
        wandbLossLog = {
            "train_loss": train_loss, "val_loss": val_loss,
            "train_pckh": train_pckh, "val_pckh": val_pckh,
            "train_mse": train_mse, "val_mse": val_mse,
            "train_ws": train_ws, "val_ws": val_ws
        }
        if WANDBLOG:
            wandb.log(wandbLossLog, step=epoch)

        print(f"Epoch {epoch:03d} ▶ Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | "
              f"Train PCKh = {train_pckh:.4f} | Val PCKh = {val_pckh:.4f}")

        # Optional topographic diagnostics/plots
        if WANDBLOG or SAVE_PLOTS:
            pref_t = topo_analysis.plot_keypoint_parcellation_from_head(
                model, title="Keypoint Parcellation", grid_hw=TOPO_GRID, topo=True, epoch=epoch,
                wandblog=WANDBLOG, save_plots=SAVE_PLOTS, save_dir=metrics_dir
            )

            topo_analysis.plot_coactivation_distance_curve(
                model, val_loader, device, topo=True, grid_hw=TOPO_GRID, epoch=epoch,
                wandblog=WANDBLOG, save_plots=SAVE_PLOTS, save_dir=metrics_dir
            )

            topo_analysis.plot_activation_neighbor_corr(
                model.topo, model, val_loader, device, topo=True,
                title_prefix="Classic activation neighbor corr", epoch=epoch,
                wandblog=WANDBLOG, save_plots=SAVE_PLOTS, save_dir=metrics_dir
            )

            # Scalar topographic metrics
            I_t, _ = topo_analysis.morans_I_strength(model, topo=True, grid_hw=TOPO_GRID)
            P_c = topo_analysis.parcellation_contiguity(pref_t)

            if WANDBLOG:
                wandb.log({"metrics/moransI_strength/topo": I_t}, step=epoch)
                wandb.log({"metrics/contiguity/topo": P_c}, step=epoch)

            if SAVE_PLOTS:
                with open(os.path.join(metrics_dir, "metrics.csv"), "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, train_loss, val_loss, train_pckh, val_pckh,
                                     train_mse, val_mse, train_ws, val_ws, I_t, P_c])

        # Checkpoint best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, run_name + ".pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model to {save_path}")

    if WANDBLOG:
        wandb.finish()


if __name__ == '__main__':
    main()
