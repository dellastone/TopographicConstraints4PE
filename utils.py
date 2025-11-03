# utils.py

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import wandb
def _wandb_fig(image, caption=None):
    return wandb.Image(image, caption=caption) if caption else wandb.Image(image)
class EarlyStopping:
    def __init__(self, patience=3, verbose=True, name="null"):
        self.patience = patience
        self.verbose = verbose
        self.name = name
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
            
def extract_keypoints_from_heatmaps(heatmaps):
    """
    Extract (x, y) coordinates from heatmaps using argmax.
    Returns normalized coordinates in [0, 1].
    Args:
        heatmaps: (B, K, H, W)
    Returns:
        keypoints: (B, K, 2) - normalized x, y
    """
    B, K, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.view(B, K, -1)
    max_indices = heatmaps_reshaped.argmax(dim=2)  # (B, K)

    y = (max_indices // W).float()
    x = (max_indices % W).float()

    x_norm = x / W
    y_norm = y / H

    return torch.stack([x_norm, y_norm], dim=2)  # (B, K, 2)

def compute_pckh_from_heatmaps(pred_heatmaps, gt_heatmaps, head_indices=(9, 12), threshold=0.5, visibility=None):
    """
    Compute PCKh metric from predicted and GT heatmaps.
    
    Args:
        pred_heatmaps (Tensor): (B, K, H, W) predicted heatmaps
        gt_heatmaps (Tensor): (B, K, H, W) ground truth heatmaps
        head_indices (tuple): keypoints to use for head size (e.g. (9, 12))
        threshold (float): threshold relative to head size
        visibility (Tensor or None): (B, K) optional visibility mask
    
    Returns:
        pckh_per_kpt (Tensor): (K,) average PCKh per keypoint
        mean_pckh (float): overall average PCKh
    """
    # Get predicted and true keypoints from heatmaps
    pred_kpts = extract_keypoints_from_heatmaps(pred_heatmaps)
    gt_kpts = extract_keypoints_from_heatmaps(gt_heatmaps)

    B, K, _ = pred_kpts.shape

    # Compute head size: distance between two GT head keypoints
    head_a = gt_kpts[:, head_indices[0]]
    head_b = gt_kpts[:, head_indices[1]]
    head_size = torch.norm(head_a - head_b, dim=1).clamp(min=1e-6).unsqueeze(1)  # (B, 1)

    # Compute distances between predicted and GT keypoints
    dists = torch.norm(pred_kpts - gt_kpts, dim=2)  # (B, K)

    # Compute correct predictions within threshold × head size
    correct = (dists <= threshold * head_size)  # (B, K)

    if visibility is not None:
        correct = correct & (visibility > 0)

    # Average per keypoint and overall
    pckh_per_kpt = correct.float().mean(dim=0)  # (K,)
    mean_pckh = pckh_per_kpt.mean().item()

    return pckh_per_kpt, mean_pckh


def plot_image_with_heatmaps(image_tensor: torch.Tensor,
                             heatmaps: torch.Tensor,
                             radius: int = 3,
                             color: tuple = (255, 0, 0),
                             wandblog: bool = False,
                             save_plots: bool = False, save_dir: str = None):
    """
    Plots the original image with keypoints extracted from heatmaps.

    Args:
        image_tensor (Tensor): Normalized image tensor (3, H, W).
        heatmaps     (Tensor): Heatmaps of shape (K, Hh, Wh).
        radius       (int):   Radius of the keypoint circles.
        color        (tuple): BGR color of keypoints in OpenCV format.
        pause        (float): Seconds to pause after drawing.
    """
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(3,1,1)
    img = image_tensor * std + mean                # (3, H, W)
    img = img.permute(1,2,0).cpu().numpy()         # (H, W, 3)
    img = (img * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    H, W = img.shape[:2]
    K, Hh, Wh = heatmaps.shape

    for k in range(K):
        with torch.no_grad():
            # Convert heatmap to numpy and move to CPU
            hm = heatmaps[k].cpu().numpy()
            # Resize heatmap to image size
            hm_resized = cv2.resize(hm, (W, H))
            # Find peak
            y, x = np.unravel_index(np.argmax(hm_resized), hm_resized.shape)
            cv2.circle(img_bgr, (x, y), radius, color, thickness=-1)

    # Display with matplotlib
    fig = plt.figure(figsize=(6,6 * H/W))
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Predicted Keypoints Overlay")
    figs = {}
    figs[f'Predicted Keypoints Overlay'] = fig
    wandb_images = { f"plots/{k}": _wandb_fig(figs[k]) for k in figs }
    if wandblog:
        wandb.log(wandb_images)
    if save_plots:
        fig_path = f"{save_dir}/predicted_keypoints_overlay.png"
        fig.savefig(fig_path)
    # plt.pause(pause)
    plt.close(fig)



