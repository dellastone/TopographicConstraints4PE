# topographic.py
import torch
import torch.nn as nn
import math

def build_edges(H, W, mode="8"):
    """
    Undirected neighbor pairs across a Gh×Gw channel grid; each pair counted once (a<b).
    mode: "4" (N,S,E,W) or "8" (adds diagonals).
    """
    nbrs4 = [(1,0),(-1,0),(0,1),(0,-1)]
    nbrs8 = nbrs4 + [(1,1),(1,-1),(-1,1),(-1,-1)]
    nbrs = nbrs4 if mode == "4" else nbrs8
    edges = []
    for i in range(H):
        for j in range(W):
            a = i*W + j
            for di,dj in nbrs:
                ni, nj = i+di, j+dj
                if 0 <= ni < H and 0 <= nj < W:
                    b = ni*W + nj
                    if a < b: edges.append((a,b))
    if not edges:
        return torch.empty(0, 2, dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long)

class TopographicConv2d(nn.Module):
    def __init__(self, in_channels, grid_h=16, grid_w=16,
                 neighbors="8", use_bn=False, bn_affine=False, use_relu=True):
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w
        self.conv = nn.Conv2d(in_channels, grid_h*grid_w, kernel_size=1, bias = not use_bn)
        self.bn = nn.BatchNorm2d(grid_h*grid_w, affine=bn_affine) if use_bn else None
        self.relu = nn.ReLU(inplace=True) if use_relu else None
        self.register_buffer("edge_index", build_edges(grid_h, grid_w, neighbors), persistent=False)
        self._last_activation = None
        
        self.reset_parameters(method="orthogonal")


    @property
    def out_channels(self): return self.grid_h * self.grid_w

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None: y = self.bn(y)
        if self.relu is not None: y = self.relu(y)
        self._last_activation = y  # (B, C, H, W)
        return y

    def weight_similarity_loss(self):

        if self.edge_index.numel() == 0:
            return self.conv.weight.new_zeros(())

        W = self.conv.weight.view(self.out_channels, -1) 

        u, v = self.edge_index[:,0], self.edge_index[:,1]
        d2 = (W[u] - W[v]).pow(2).sum(dim=1)  
        return d2.mean()
    
    def reset_parameters(self, method: str = "orthogonal"):
        """
        method ∈ {"orthogonal","kaiming","equalnorm"}.
        - "orthogonal": semi-orthogonal rows with ReLU gain (default; robust for WS).
        - "kaiming": Kaiming normal for ReLU.
        - "equalnorm": Kaiming, then row-normalize to equal L2 norm.
        """
        Cout = self.out_channels
        fan_in = self.conv.in_channels  # 1x1 kernel

        # ---- Conv weights ----
        if method == "orthogonal":
            w = self.conv.weight.data.view(Cout, -1)
            gain = nn.init.calculate_gain('relu') if self.relu is not None else 1.0
            nn.init.orthogonal_(w, gain=gain)     # semi-orthogonal rows
            self.conv.weight.data.copy_(w.view_as(self.conv.weight))
        elif method == "kaiming":
            nn.init.kaiming_normal_(
                self.conv.weight,
                mode='fan_in',
                nonlinearity='relu' if self.relu is not None else 'linear'
            )
        elif method == "equalnorm":
            nn.init.kaiming_normal_(
                self.conv.weight,
                mode='fan_in',
                nonlinearity='relu' if self.relu is not None else 'linear'
            )
            w = self.conv.weight.data.view(Cout, -1)
            w = w / (w.norm(dim=1, keepdim=True) + 1e-9)
            # set a common scale consistent with ReLU
            scale = math.sqrt(2.0 / fan_in) if self.relu is not None else 1.0 / math.sqrt(fan_in)
            w.mul_(scale)
            self.conv.weight.data.copy_(w.view_as(self.conv.weight))
        else:
            raise ValueError(f"Unknown init method: {method}")

        # Bias: zero (and usually disabled if BN is present)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        if self.bn is not None:
            self.bn.reset_running_stats()
            if getattr(self.bn, "affine", False):
                nn.init.ones_(self.bn.weight)
                nn.init.zeros_(self.bn.bias)
