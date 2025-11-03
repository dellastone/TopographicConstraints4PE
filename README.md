# Topographic Pose Estimation – README

This repository contains code for training and analyzing human pose estimation models on MPII. It includes:

- **SimpleBaseline**: a standard ResNet backbone + deconvolution head that predicts keypoint heatmaps.
- **TopoModel**: extends SimpleBaseline with a **TopographicConv2d** layer that arranges channels on a 2D grid and adds a spatial regularization loss encouraging topographic organization.

The code also provides utilities for loading MPII annotations, creating heatmaps, computing **PCKh**, visualizing predictions, and analyzing topographic structure (e.g., neighbor correlations and Moran’s I).

---

## 1) Environment & Installation

Tested with Python 3.10+.

```bash
# (Recommended) create a fresh environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # pick the CUDA version you have
pip install numpy scipy opencv-python matplotlib tqdm wandb
```

> If you are on CPU-only, install the CPU wheels for torch/torchvision accordingly.

---

## 2) Dataset: MPII Human Pose

This project expects **MPII** images and the official annotation `.mat` file.

- **Annotations**: `mpii_human_pose_v1_u12_1.mat`
- **Images directory**: a folder containing the MPII images (`.jpg`)

Suggested layout:
```
project/
  mpii/
    images/
      000000001.jpg
      000000002.jpg
      ...
    mpii_human_pose_v1_u12_1.mat
```

By default the scripts look for:
- `--mat-path ./mpii/mpii_human_pose_v1_u12_1.mat`
- `--img-dir  ./mpii/images`

You can change these via CLI flags.

---

## 3) Project Structure

```
dataset.py              # MPII parsing + heatmap generation + Dataset
model.py                # SimpleBaseline + TopoModel architectures
topographic.py          # TopographicConv2d layer + spatial loss
topo_analysis.py        # Analysis/plots: parcellation, neighbor corr, Moran’s I, co-activation distance
utils.py                # PCKh metric, plotting overlay, EarlyStopping
trainSimpleBaseline.py  # Training script for SimpleBaseline
trainTopoModel.py       # Training script for TopoModel (with spatial regularization)
```

Key elements:
- **Heatmaps** are generated at `(64, 64)` by default from normalized keypoints (see `generate_heatmaps` in `dataset.py`).
- **PCKh** computation is in `utils.py` (`compute_pckh_from_heatmaps`).
- **Topographic regularization** is implemented in `TopographicConv2d.weight_similarity_loss` (see `topographic.py`), used by `TopoModel.topo_reg_loss`.

---

## 4) Quickstart

### 4.1 Train the SimpleBaseline

```bash
python trainSimpleBaseline.py \
  --device cuda \
  --batch-size 64 \
  --lr 1e-4 \
  --epochs 20 \
  --val-split 0.1 \
  --save-dir checkpoints \
  --num-keypoints 16 \
  --pretrained \
  --arch resnet152 \
  --mat-path ./mpii/mpii_human_pose_v1_u12_1.mat \
  --img-dir  ./mpii/images \
  --wandb \
  --save-plts
```

**Notable flags** (defaults in brackets):
- `--device [cuda]`  e.g., `cpu`, `cuda`, `cuda:0`
- `--batch-size [64]`, `--lr [1e-4]`, `--epochs [20]`, `--val-split [0.1]`
- `--pretrained / --no-pretrained`
- `--arch [resnet152]` (choices: resnet18/34/50/101/152)
- `--wandb` *enables* Weights & Biases logging (use `--no-wandb` to disable)
- `--save-plts` saves analysis plots under `plots/<run_name>/`

Outputs:
- **Checkpoints**: `checkpoints/<run_name>/<run_name>.pth`
- **Plots & metrics** (if `--save-plts`): `plots/<run_name>/` + `metrics.csv`
- **W&B logs** (if `--wandb`): project `ABNS`

### 4.2 Train the TopoModel

```bash
python trainTopoModel.py \
  --device cuda \
  --batch-size 64 \
  --lr 1e-4 \
  --epochs 20 \
  --val-split 0.1 \
  --save-dir checkpoints \
  --num-keypoints 16 \
  --pretrained \
  --arch resnet18 \
  --mat-path ./mpii/mpii_human_pose_v1_u12_1.mat \
  --img-dir  ./mpii/images \
  --spatial-lambda 0.5 \
  --topo-grid 16 16 \
  --wandb \
  --save-plts
```

**Additional flags**:
- `--spatial-lambda [0.5]` : base λ for the spatial (topographic) loss.
- `--topo-grid [16 16]` : grid **Gh Gw**. Aim for `Gh*Gw == 256` to match the deconv output channels.

The training script uses a cosine schedule to smoothly scale `spatial-lambda` across epochs.

---

## 5) What the scripts do

### Data loading
- `load_mpii_annotations(mat_path)` parses the MPII `.mat` file, builds a list of annotated instances, and **filters** poses with too few visible keypoints.  
- `MPIIDataset` resizes images to `(256, 256)`, rescales keypoints accordingly, normalizes images, and creates `(num_keypoints, 64, 64)` heatmaps.

### Models
- **SimpleBaseline**: ResNet backbone (C5 features) → 3× deconvs (upsampling to H/4, W/4) → 1×1 conv to `K` heatmaps.  
  Kaiming/bilinear initialization for deconvs and a small-normal head init are applied.
- **TopoModel**: same backbone/deconv, then a `TopographicConv2d` (1×1 conv producing `Gh*Gw` channels arranged on a grid) → head.  
  The **WS loss** penalizes differences between incoming weights of neighboring grid locations.

### Metrics & Visualizations
During validation (and optionally each epoch), the scripts:
- Compute **loss** and **PCKh** (overall and per-keypoint internally).
- Overlay predicted keypoints on the input image (`plot_image_with_heatmaps`).
- (When enabled) Run **topographic analyses** via `topo_analysis.py`, including:
  - **Keypoint parcellation**: dominant head weight per grid cell.
  - **Neighbor correlation** (activation-based) maps + histograms.
  - **Moran’s I** of head weights (higher = stronger spatial autocorrelation).
  - **Co-activation distance curve** across correlation thresholds.

All plots can be **logged to W&B** (`--wandb`) and/or **saved to disk** (`--save-plts`).

---

## 6) Weights & Biases (optional)

```bash
wandb login
# then run training with --wandb
```

Project name is **ABNS** by default. Each run name includes the backbone, model type, and a timestamp.

---

## 7) Tips & Troubleshooting

- **CUDA not found / version mismatch**: install the correct PyTorch wheels for your CUDA version. If in doubt, use CPU for a quick sanity check (`--device cpu`).
- **Out-of-memory**: try `--batch-size 16` (or smaller), or switch to `resnet18`.
- **Slow training**: disable W&B (`--no-wandb`) and plot saving (`--no-save-plts`), or reduce `--epochs`.
- **Missing images**: verify `--img-dir` path and that file names in the `.mat` match your images.
- **Annotation path**: ensure `--mat-path` points to the correct `.mat`.
- **Topo grid**: if you change the deconv output channels, adjust `--topo-grid` so `Gh*Gw` matches the channel count used before the head.
- **Head indices for PCKh**: `utils.compute_pckh_from_heatmaps` uses `(9, 12)` by default for head size; adapt if your keypoint indexing differs.

---

## 8) Minimal Examples

**Baseline (quick CPU test):**
```bash
python trainSimpleBaseline.py --device cpu --epochs 1 --no-wandb --no-save-plts
```

**TopoModel with weaker spatial regularization:**
```bash
python trainTopoModel.py --spatial-lambda 0.2 --topo-grid 16 16 --arch resnet34
```

---

## 9) Citing / Acknowledgements

If you use this code, please consider citing MPII and any relevant baseline/topographic modeling work.  
This repository draws on standard ResNet backbones from `torchvision` and uses common evaluation metrics (PCKh).

---

## 10) License

Add your preferred license here (e.g., MIT).