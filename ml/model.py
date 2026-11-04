"""U-Net landmark model: X-ray in, 68 heatmaps out at 1/4 resolution."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cobb import N_LANDMARKS


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class LandmarkUNet(nn.Module):
    """Encoder down to 1/32, decoder back up to 1/4, one heatmap channel per landmark."""

    def __init__(self, base: int = 32, n_out: int = N_LANDMARKS):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8, base * 16]       # 1/1 ... 1/16
        self.enc = nn.ModuleList([block(1, ch[0])] + [block(ch[i - 1], ch[i]) for i in range(1, 5)])
        self.bottom = block(ch[4], ch[4])                          # 1/32
        # decoder: 1/32 -> 1/16 -> 1/8 -> 1/4
        self.up = nn.ModuleList([
            block(ch[4] + ch[4], ch[3]),
            block(ch[3] + ch[3], ch[2]),
            block(ch[2] + ch[2], ch[1]),
        ])
        self.head = nn.Conv2d(ch[1], n_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)                                        # 1/1, 1/2, 1/4, 1/8, 1/16
        x = self.bottom(F.max_pool2d(x, 2))                        # 1/32
        for up, skip in zip(self.up, [skips[4], skips[3], skips[2]]):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = up(torch.cat([x, skip], 1))
        return self.head(x)                                        # 1/4


def soft_argmax(heatmaps: torch.Tensor, beta: float = 50.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable peak location. Returns (B x K x 2 coords at heatmap scale, B x K peak value)."""
    b, k, h, w = heatmaps.shape
    flat = heatmaps.reshape(b, k, -1)
    p = torch.softmax(flat * beta, dim=-1).reshape(b, k, h, w)
    ys = torch.arange(h, device=heatmaps.device, dtype=heatmaps.dtype)
    xs = torch.arange(w, device=heatmaps.device, dtype=heatmaps.dtype)
    x = (p.sum(2) * xs).sum(-1)
    y = (p.sum(3) * ys).sum(-1)
    peak = flat.amax(-1).clamp(0, 1)
    return torch.stack([x, y], -1), peak


def cobb_from_coords(coords: torch.Tensor) -> torch.Tensor:
    """Differentiable primary Cobb angle (degrees) from B x 68 x 2 landmarks.

    Same rule as cobb.curves_from_landmarks: the largest angle between any two endplates.
    """
    v = coords.reshape(coords.shape[0], 17, 4, 2)
    left = torch.stack([v[:, :, 0], v[:, :, 2]], 2).reshape(-1, 34, 2)    # TL, BL
    right = torch.stack([v[:, :, 1], v[:, :, 3]], 2).reshape(-1, 34, 2)   # TR, BR
    d = right - left
    t = torch.rad2deg(torch.atan2(d[..., 1], d[..., 0]))
    t = torch.remainder(t + 90, 180) - 90                                 # wrap to [-90, 90) like cobb.tilt_deg
    diff = (t[:, :, None] - t[:, None, :]).abs()
    diff = torch.minimum(diff, 180 - diff)                                # acute angle, like cobb.angle_between
    return diff.reshape(diff.shape[0], -1).amax(-1)
