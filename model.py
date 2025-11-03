# model.py

import torch.nn as nn
import torchvision.models as models
from topographic import TopographicConv2d
import torch
_RESNET_OUT_CHANNELS = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
}

def _bilinear_kernel(k: int):
    """2D bilinear upsampling kernel (k×k) as a torch.Tensor."""
    factor = (k + 1) // 2
    center = factor - 1 if k % 2 == 1 else factor - 0.5
    og = torch.arange(k, dtype=torch.float32)
    filt = (1 - torch.abs(og - center) / factor).unsqueeze(0) * \
           (1 - torch.abs(og - center) / factor).unsqueeze(1)
    return filt

def init_simplebaseline(self, *, use_bilinear_for_equal_256=True):
    """
    Initialize deconvs + head.
    - Deconvs: Kaiming normal (ReLU). Optionally bilinear init on the two 256→256 ConvT.
    - BN: weight=1, bias=0
    - Head: Normal(0, 0.001), bias=0
    """
    # ---- Deconv stack ----
    for m in self.deconv.modules():
        if isinstance(m, nn.ConvTranspose2d):
            k = m.kernel_size[0]
            if use_bilinear_for_equal_256 and m.in_channels == m.out_channels == 256 and k == 4:
                # Diagonal bilinear init (upsampling-like start)
                w = torch.zeros_like(m.weight)
                bil = _bilinear_kernel(k).to(w.dtype).to(w.device)
                d = min(m.out_channels, m.in_channels)
                for i in range(d):
                    w[i, i, :, :] = bil
                with torch.no_grad():
                    m.weight.copy_(w)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ---- Head (1×1 conv to K heatmaps) ----
    nn.init.normal_(self.head.weight, std=0.001)
    if self.head.bias is not None:
        nn.init.zeros_(self.head.bias)

class SimpleBaseline(nn.Module):
    """
    SimpleBaseline:
      <ResNet backbone> → C5 → 3× deconv → k heatmaps.

    Args:
        num_keypoints (int): number of predicted heatmaps.
        backbone_name (str): one of 'resnet18','resnet34','resnet50','resnet101','resnet152'.
        pretrained (bool): whether to load ImageNet weights for the backbone.
    """
    def __init__(self,
                 num_keypoints: int = 16,
                 backbone_name: str = 'resnet18',
                 pretrained: bool = False):
        super().__init__()

        assert backbone_name in _RESNET_OUT_CHANNELS, \
            f"Unknown backbone '{backbone_name}'"
        weights = None
        if pretrained:
            # Use the new weights enum API if available
            try:
                weights_enum = getattr(models, f"{backbone_name.capitalize()}_Weights").DEFAULT
                weights = weights_enum
            except AttributeError:
                weights = 'IMAGENET1K_V1'  # fallback for older torchvision
        resnet = getattr(models, backbone_name)(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        c5_channels = _RESNET_OUT_CHANNELS[backbone_name]


        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c5_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        

        self.head = nn.Conv2d(256, num_keypoints, kernel_size=1)
        
        init_simplebaseline(self)

    def forward(self, x):
        x = self.backbone(x)  # → (B, C5, H/32, W/32)
        x = self.deconv(x)    # → (B, 256, H/4, W/4)
        return self.head(x)   # → (B, K, H/4, W/4)


class TopoModel(nn.Module):
    def __init__(self,
                 num_keypoints: int = 16,
                 backbone_name: str = 'resnet18',
                 pretrained: bool = False,
                 topo_grid_h: int = 16,
                 topo_grid_w: int = 16):
        super().__init__()
        assert backbone_name in _RESNET_OUT_CHANNELS
        weights = None
        if pretrained:
            try:
                weights = getattr(models, f"{backbone_name.capitalize()}_Weights").DEFAULT
            except AttributeError:
                weights = 'IMAGENET1K_V1'
        resnet = getattr(models, backbone_name)(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # C5
        c5_channels = _RESNET_OUT_CHANNELS[backbone_name]

        self.deconv = nn.Sequential(
                            nn.ConvTranspose2d(c5_channels, 256, kernel_size=4, stride=2, padding=1), 
                            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                            nn.ConvTranspose2d(256, 256, 4, 2, 1),
                            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                            nn.ConvTranspose2d(256, 256, 4, 2, 1),
                            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                        )
        self.topo = TopographicConv2d(in_channels=256, grid_h=topo_grid_h, grid_w=topo_grid_w, use_bn=False)
        self.head = nn.Conv2d(self.topo.out_channels, num_keypoints, kernel_size=1)
        init_simplebaseline(self)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.deconv(x)
        x = self.topo(x)
        return self.head(x)
    
    def topo_reg_loss(self, lambda_ws=0.0, use_bn_effective=False):
        ws = self.topo.weight_similarity_loss(use_bn_effective=use_bn_effective)
        return lambda_ws * ws
