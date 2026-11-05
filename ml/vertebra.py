"""Vertebra-centre model: find the 17 vertebra centres, then each one's 4 corners.

Predicting 68 separate corner heatmaps from scratch is hard to learn on ~400
X-rays (every channel is almost all background). Instead, following the
approach of the strongest AASCE solutions (Yi et al., MICCAI 2020), the network
outputs at 1/4 resolution:

  * 1 heatmap of vertebra centres (one class, 17 positives per image),
  * 2 channels: sub-pixel offset of the centre,
  * 8 channels: vectors from the centre to the TL, TR, BL, BR corners.

Decoding takes the 17 strongest local peaks, sorts them top to bottom and adds
the corner vectors, giving the usual 68 landmarks (see cobb.py for the order).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cobb import N_VERTEBRAE
from model import LandmarkUNet

STRIDE = 4
HM_SIGMA = 1.5          # centre Gaussian, heatmap pixels
VEC_SCALE = 0.1         # corner vectors are regressed in units of 10 heatmap px


class VertebraNet(nn.Module):
    def __init__(self, base: int = 16):
        super().__init__()
        self.net = LandmarkUNet(base=base, n_out=1 + 2 + 8)
        with torch.no_grad():
            self.net.head.bias[0] = -2.19                  # CenterNet prior: start at p = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def targets(pts_in: np.ndarray, h: int, w: int):
    """Training targets from 68 x 2 landmarks in input pixels."""
    v = pts_in.reshape(N_VERTEBRAE, 4, 2) / STRIDE
    centres = v.mean(1)                                     # 17 x 2, heatmap px
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    hm = np.zeros((1, h, w), np.float32)
    ind = np.zeros(N_VERTEBRAE, np.int64)
    reg = np.zeros((N_VERTEBRAE, 10), np.float32)
    valid = np.zeros(N_VERTEBRAE, np.float32)
    for k, (cx, cy) in enumerate(centres):
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        g = np.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2 * HM_SIGMA ** 2))
        hm[0] = np.maximum(hm[0], g)
        ind[k] = iy * w + ix
        reg[k, :2] = cx - ix, cy - iy
        reg[k, 2:] = ((v[k] - centres[k]) * VEC_SCALE).reshape(-1)
        valid[k] = 1
    return hm, ind, reg, valid


def focal_loss(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = gt.ge(0.999).float()
    pos_l = torch.log(p) * (1 - p) ** 2 * pos
    neg_l = torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * (1 - pos)
    return -(pos_l.sum() + neg_l.sum()) / pos.sum().clamp(min=1)


def loss_fn(out: torch.Tensor, hm, ind, reg, valid, vec_weight: float = 1.0):
    b, c, h, w = out.shape
    l_hm = focal_loss(out[:, :1], hm)
    feat = out[:, 1:].reshape(b, 10, h * w)
    pred = torch.gather(feat, 2, ind[:, None, :].expand(b, 10, ind.shape[1])).permute(0, 2, 1)   # b x 17 x 10
    m = valid[..., None]
    n = m.sum().clamp(min=1)
    l_off = (F.l1_loss(pred[..., :2], reg[..., :2], reduction="none") * m).sum() / n
    l_vec = (F.l1_loss(pred[..., 2:], reg[..., 2:], reduction="none") * m).sum() / n
    return l_hm + l_off + vec_weight * l_vec, (l_hm.item(), l_off.item(), l_vec.item())


def decode(out: torch.Tensor):
    """B x 11 x h x w network output -> (B x 68 x 2 landmarks in input px, B x 17 scores).

    Pure tensor ops (max-pool NMS, top-k, gather, sort) so it exports to ONNX.
    """
    b, _, h, w = out.shape
    heat = torch.sigmoid(out[:, :1])
    keep = (F.max_pool2d(heat, 3, 1, 1) == heat).float()
    scores, idx = torch.topk((heat * keep).reshape(b, -1), N_VERTEBRAE)            # b x 17
    feat = out[:, 1:].reshape(b, 10, h * w)
    reg = torch.gather(feat, 2, idx[:, None, :].expand(b, 10, N_VERTEBRAE)).permute(0, 2, 1)
    cx = (idx % w).float() + reg[..., 0]
    cy = torch.div(idx, w, rounding_mode="floor").float() + reg[..., 1]
    order = torch.argsort(cy, dim=1)                                                 # top to bottom
    cx, cy = torch.gather(cx, 1, order), torch.gather(cy, 1, order)
    vec = torch.gather(reg[..., 2:], 1, order[..., None].expand(b, N_VERTEBRAE, 8)).reshape(b, N_VERTEBRAE, 4, 2)
    corners = torch.stack([cx, cy], -1)[:, :, None, :] + vec / VEC_SCALE
    return corners.reshape(b, N_VERTEBRAE * 4, 2) * STRIDE, torch.gather(scores, 1, order)
