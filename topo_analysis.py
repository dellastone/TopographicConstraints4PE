import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from utils import _wandb_fig


def _auto_channel_grid(C: int):
    """Pick a near-square H×W grid for C channels (e.g., 256 → 16×16)."""
    h = int(C ** 0.5)
    while C % h != 0:
        h -= 1
    return h, C // h


def _get_grid_hw_from_model_or_auto(model, C, topo: bool, grid_hw):
    if topo:
        Gh, Gw = int(model.topo.grid_h), int(model.topo.grid_w)
        assert Gh * Gw == C, f"Topo grid {Gh}x{Gw}!={C}"
        return Gh, Gw
    if grid_hw is not None:
        Gh, Gw = grid_hw
        assert Gh * Gw == C, f"grid_hw {grid_hw} != {C} channels"
        return Gh, Gw
    return _auto_channel_grid(C)


def _neighbor_pairs(Gh, Gw, kind='8'):
    nbrs = []
    for r in range(Gh):
        for c in range(Gw):
            i = r*Gw+c
            if kind == '4':
                offsets = [(-1,0),(1,0),(0,-1),(0,1)]
            else:
                offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
            for dr,dc in offsets:
                rr,cc = r+dr, c+dc
                if 0 <= rr < Gh and 0 <= cc < Gw:
                    j = rr*Gw+cc
                    if i < j:
                        nbrs.append((i,j))
    return nbrs



@torch.no_grad()
def plot_keypoint_parcellation_from_head(model, title="Parcellation by |head weights|", grid_hw=None, topo: bool = False, epoch: int = None, 
                                          wandblog: bool = True, save_plots: bool = False, save_dir: str = None):
    

    # Outgoing weights (K, C, 1, 1) -> (K, C)
    W = model.head.weight.detach()[:, :, 0, 0]   # [K, C]
    K, C = W.shape

    if topo:
        Gh, Gw = int(model.topo.grid_h), int(model.topo.grid_w)
        assert Gh * Gw == C, f"Topo grid {Gh}x{Gw}!={C} channels"
    else:
        if grid_hw is None:
            Gh, Gw = _auto_channel_grid(C)
        else:
            Gh, Gw = grid_hw
            assert Gh * Gw == C, f"grid_hw {grid_hw} != {C} channels"

    pref = W.abs().argmax(dim=0).view(Gh, Gw).cpu().numpy()  # (Gh, Gw)

    # Plot
    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(pref, interpolation='nearest', cmap='tab20')
    cbar = plt.colorbar(im)
    cbar.set_label('Keypoint id')
    plt.title(title)
    plt.xlabel('Grid W'); plt.ylabel('Grid H')
    plt.tight_layout()
    figs = {}
    figs[f'keypoint_parcellation_from_head'] = fig
    if wandblog:
        wandb_images = { f"plots/{k}": _wandb_fig(figs[k]) for k in figs }
        wandb.log(wandb_images, step=epoch)
    if save_plots:
        fig_path = f"{save_dir}/keypoint_parcellation_from_head_epoch_{epoch if epoch is not None else 'final'}.png"
        fig.savefig(fig_path)
    plt.close(fig)
    
    return pref


@torch.no_grad()
def plot_activation_neighbor_corr(module, model, loader, device, *,
                                  topo=False, grid_hw=None, max_batches=10, neighbor='8',
                                  title_prefix="Neighbor corr (activations)", epoch=None,
                                  wandb_tag_prefix="plots/activation_corr",
                                  wandblog=False, save_plots=False, save_dir=None):
    feats = []
    h = module.register_forward_hook(lambda m, inp, out: feats.append(out.detach()))
    for b,(images, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        _ = model(images)
        if b+1 >= max_batches: break
    h.remove()
    if not feats: return

    F = torch.cat(feats, 0)                 # [N,C,H',W']
    A = F.mean(dim=(2,3))                   # [N,C]
    A = (A - A.mean(dim=1, keepdim=True)) / (A.std(dim=1, keepdim=True) + 1e-6)
    A = (A - A.mean(dim=0, keepdim=True)) / (A.std(dim=0, keepdim=True) + 1e-6)
    Cmat = torch.clamp((A.T @ A) / max(1, A.shape[0]-1), -1, 1).cpu().numpy()

    C = Cmat.shape[0]
    Gh, Gw = _get_grid_hw_from_model_or_auto(model, C, topo, grid_hw)
    nbrs = _neighbor_pairs(Gh, Gw, kind=neighbor)

    # mean neighbor corr per channel
    vals = np.zeros(C, dtype=np.float32)
    cnts = np.zeros(C, dtype=np.int32)
    for i,j in nbrs:
        vals[i] += Cmat[i,j]; cnts[i]+=1
        vals[j] += Cmat[i,j]; cnts[j]+=1
    cnts = np.maximum(cnts, 1)
    mean_map = (vals / cnts).reshape(Gh, Gw)

    # map plot
    fig_m = plt.figure(figsize=(5,5))
    plt.imshow(mean_map, vmin=-1, vmax=1, interpolation='nearest')
    plt.colorbar(label='mean neighbor corr')
    plt.title(f"{title_prefix}: mean map"); plt.xlabel('Grid W'); plt.ylabel('Grid H')
    plt.tight_layout()

    # histogram
    nbr_vals = [float(Cmat[i,j]) for i,j in nbrs]
    fig_h = plt.figure(figsize=(6,4))
    bins = np.linspace(-1, 1, 41)
    plt.hist(nbr_vals, bins=bins, alpha=0.8, density=True)
    plt.xlabel('neighbor correlation'); plt.ylabel('density')
    plt.title(f"{title_prefix}: histogram")
    plt.tight_layout()

    if wandblog:
        wandb.log({
            f"{wandb_tag_prefix}/map": _wandb_fig(fig_m),
            f"{wandb_tag_prefix}/hist": _wandb_fig(fig_h),
        }, step=epoch)
    if save_plots:
        fig_path_m = f"{save_dir}/activation_neighbor_corr_map_epoch_{epoch if epoch is not None else 'final'}.png"
        fig_m.savefig(fig_path_m)
        fig_path_h = f"{save_dir}/activation_neighbor_corr_hist_epoch_{epoch if epoch is not None else 'final'}.png"
        fig_h.savefig(fig_path_h)

    plt.close(fig_m); plt.close(fig_h)


def parcellation_contiguity(pref):
    Gh,Gw = pref.shape; same=total=0
    for r in range(Gh):
        for c in range(Gw):
            if r+1<Gh: total+=1; same += int(pref[r,c]==pref[r+1,c])
            if c+1<Gw: total+=1; same += int(pref[r,c]==pref[r,c+1])
    return same/max(1,total)


def morans_I_strength(model, topo=False, grid_hw=None):
    K = model.head.out_channels
    C = model.head.weight.shape[1]
    Gh,Gw = _get_grid_hw_from_model_or_auto(model, C, topo, grid_hw)
    nbrs = _neighbor_pairs(Gh, Gw, kind='8')
    W = np.zeros((Gh*Gw, Gh*Gw), dtype=np.float32)
    for i,j in nbrs: W[i,j]=W[j,i]=1
    Wsum = W.sum()

    I_vals=[]
    with torch.no_grad():
        for k in range(K):
            x = model.head.weight[k, :, 0, 0].abs().view(Gh, Gw).flatten().cpu().numpy()
            x = (x - x.mean()); denom = (x**2).sum()+1e-12
            num = 0.0
            for i,j in nbrs: num += (x[i]*x[j])*2  
            I = (Gh*Gw / Wsum) * (num / denom)
            I_vals.append(I)
    return float(np.mean(I_vals)), float(np.std(I_vals))

@torch.no_grad()
def plot_coactivation_distance_curve(model, loader, device, *,
                                     topo=False, grid_hw=None, epoch=None,
                                     max_batches=16, alphas=(0.1,0.3,0.5,0.7,0.9),
                                     wandb_tag="plots/topo/coact_distance",
                                     wandblog=False, save_plots=False, save_dir=None):
    feats = []
    hook_layer = getattr(model, "topo", None) if topo else model.deconv
    h = hook_layer.register_forward_hook(lambda m, i, o: feats.append(o.detach()))
    for b,(x,_) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        _ = model(x)
        if b+1 >= max_batches: break
    h.remove()
    if not feats: return

    F = torch.cat(feats, 0)                       # [N,C,H',W']
    A = F.mean(dim=(2,3))                         # [N,C]
    A = (A - A.mean(0, keepdim=True)) / (A.std(0, keepdim=True)+1e-6)
    C = A.shape[1]

    # Channel grid
    def _auto_channel_grid(C):
        h=int(C**0.5)
        while C%h!=0: h-=1
        return h, C//h
    if topo:
        Gh, Gw = int(model.topo.grid_h), int(model.topo.grid_w)
    else:
        Gh, Gw = _auto_channel_grid(C) if grid_hw is None else grid_hw

    Cmat = torch.clamp((A.T @ A) / max(1, A.shape[0]-1), -1, 1).cpu().numpy()

    # For each alpha: mean grid distance of pairs with r>=alpha
    import numpy as np, matplotlib.pyplot as plt
    coords = np.stack(np.meshgrid(np.arange(Gh), np.arange(Gw), indexing="ij"), -1).reshape(-1,2)
    dists = np.sqrt(((coords[:,None,:]-coords[None,:,:])**2).sum(-1))  # [C,C]

    xs, ys = [], []
    for a in alphas:
        mask = (Cmat >= a) & (~np.eye(C, dtype=bool))
        if mask.sum() == 0:
            xs.append(a); ys.append(np.nan); continue
        mean_dist = float(dists[mask].mean())
        xs.append(a); ys.append(mean_dist)

    fig = plt.figure(figsize=(5,4))
    plt.plot(xs, ys, marker='o')
    plt.xlabel("Correlation threshold α"); plt.ylabel("Mean distance between co-activated pairs")
    plt.title("Co-activation spatial clustering")
    plt.grid(True); plt.tight_layout()
    if wandblog:
        wandb.log({f"{wandb_tag}": wandb.Image(fig)}, step=epoch)
    if save_plots:
        fig_path = f"{save_dir}/coactivation_distance_curve_epoch_{epoch if epoch is not None else 'final'}.png"
        fig.savefig(fig_path)
    plt.close(fig)
