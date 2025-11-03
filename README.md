# Topographic Pose Estimation – README

This repository contains code for training and analyzing human pose estimation models on MPII. It includes:

- **SimpleBaseline**: a standard ResNet backbone + deconvolution head that predicts keypoint heatmaps.
- **TopoModel**: extends SimpleBaseline with a **TopographicConv2d** layer that arranges channels on a 2D grid and adds a spatial regularization loss encouraging topographic organization.

The code also provides utilities for loading MPII annotations, creating heatmaps, computing **PCKh**, visualizing predictions, and analyzing topographic structure (e.g., neighbor correlations and Moran’s I).

---

## Environment & Installation

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

##  Dataset: MPII Human Pose
**Shortcut:** You can download a pre-organized MPII folder with the correct structure here: [Google Drive link](https://drive.google.com/drive/folders/14PblxOBLduTq2CtRAGioKuL0o6W0W3-d?usp=sharing)

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

## Project Structure

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

##  Minimal Examples

**Baseline:**
```bash
python trainSimpleBaseline.py --arch resnet18 --no-wandb --save-plts
```

**TopoModel:**
```bash
python trainTopoModel.py --arch resnet34 --no-wandb --save-plts
```

---


## Weights & Biases (optional)

```bash
wandb login
# then run training with --wandb
```

Project name is **ABNS** by default. Each run name includes the backbone, model type, and a timestamp.

---


