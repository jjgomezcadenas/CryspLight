"""Models for (x,y,z) reconstruction from the 8x8 light map.

CryspNet: modern small ConvNet (AlexNet lineage via VGG/ResNet -- 3x3 convs,
BatchNorm, residual blocks, global average pooling) sized to an 8x8 input.
MLP: flat-pixel baseline. anger_xy / fit_anger_z: classical Anger baseline.

Inputs everywhere: map normalized to sum 1 (the shape) + log(npe) scalar
(standardized) so the design survives variable-energy events later.
"""
import numpy as np
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return self.act(x + h)


class CryspNet(nn.Module):
    """1x8x8 (+ scalar) -> (x, y, z) in normalized [-1, 1] coordinates."""

    def __init__(self, in_ch=1, n_scalar=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.stage1 = nn.Sequential(ResBlock(64), ResBlock(64))       # 8x8
        self.down = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))               # 4x4
        self.stage2 = nn.Sequential(ResBlock(128), ResBlock(128))
        self.head = nn.Sequential(
            nn.Linear(128 + n_scalar, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 3))

    def forward(self, m, s):
        h = self.stage2(self.down(self.stage1(self.stem(m))))
        h = h.mean(dim=(2, 3))                                        # GAP
        return self.head(torch.cat([h, s], dim=1))


class MLP(nn.Module):
    """Flat 64-pixel baseline: same inputs, no convolutional prior."""

    def __init__(self, n_pix=64, n_scalar=1, width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_pix + n_scalar, width), nn.ReLU(inplace=True),
            nn.Linear(width, width), nn.ReLU(inplace=True),
            nn.Linear(width, 3))

    def forward(self, m, s):
        return self.net(torch.cat([m.flatten(1), s], dim=1))


# ---- classical Anger baseline -------------------------------------------------

def anger_moments(maps, pitch=6.0):
    """Centroid (x, y) [mm] and RMS radius [mm] of (n, ny, nx) maps."""
    w = maps.astype(np.float64)
    s = w.sum(axis=(1, 2))
    ii = (np.arange(maps.shape[2]) + 0.5) * pitch
    jj = (np.arange(maps.shape[1]) + 0.5) * pitch
    xc = (w.sum(axis=1) @ ii) / s
    yc = (w.sum(axis=2) @ jj) / s
    dx2 = (w.sum(axis=1) @ ii**2) / s - xc**2
    dy2 = (w.sum(axis=2) @ jj**2) / s - yc**2
    return xc, yc, np.sqrt(np.maximum(dx2 + dy2, 0.0))


def fit_anger_z(r_train, z_train, deg=3):
    """Map RMS radius -> z by polynomial regression on the training set."""
    coef = np.polyfit(r_train, z_train, deg)
    return lambda r: np.polyval(coef, r)
