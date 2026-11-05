"""Run a trained model on one image and return landmarks, confidences and curves."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cobb import curves_from_landmarks
from dataset import STRIDE, letterbox, to_array
from model import LandmarkUNet, soft_argmax
from vertebra import VertebraNet, decode

LOW_CONF_MEAN = 0.3      # mean peak height below this -> ask the user to check
COLLAPSE_PX = 3.0        # all 4 corners of a vertebra within this many px -> suspicious


class Predictor:
    def __init__(self, weights: str | Path, device: str = "cpu"):
        ckpt = torch.load(weights, map_location=device, weights_only=False)  # our own checkpoints
        self.version = ckpt.get("version", "unet-v0.1")
        self.kind = ckpt.get("kind", "heatmap")
        net = VertebraNet(base=ckpt.get("base", 16)) if self.kind == "vertebra" else LandmarkUNet(base=ckpt.get("base", 32))
        self.model = net.to(device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.device = device

    @torch.no_grad()
    def __call__(self, img: Image.Image) -> dict:
        boxed, lb = letterbox(img)
        x = torch.from_numpy(to_array(boxed))[None].to(self.device)
        out = self.model(x)
        if self.kind == "vertebra":
            lm, peak = decode(out)
            pts_in = lm[0].cpu().numpy()
        else:
            coords, peak = soft_argmax(out)
            pts_in = coords[0].cpu().numpy() * STRIDE
        pts = lb.to_original(pts_in)
        conf = peak[0].cpu().numpy()

        spread = np.ptp(pts.reshape(17, 4, 2), axis=1).max(axis=1)
        low = bool(conf.mean() < LOW_CONF_MEAN or (spread < COLLAPSE_PX).any())
        return {
            "model_version": self.version,
            "image_size": {"width": img.width, "height": img.height},
            "landmarks": np.round(pts, 1).tolist(),
            "landmark_confidence": np.round(conf, 3).tolist(),
            "curves": [c.to_dict() for c in curves_from_landmarks(pts)],
            "low_confidence": low,
            "disclaimer": "Research prototype. Not a diagnosis.",
        }


def spine_box(landmarks: np.ndarray, width: int, height: int) -> tuple[float, float, float, float]:
    """Crop box (x, y, w, h) framing the detected spine from just above T1 to just below L5,
    with the width a film of that height would have. Shared rule with web/index.html."""
    lm = np.asarray(landmarks, float)
    y0, y1 = np.percentile(lm[:, 1], [2, 98])
    pad = 0.08 * (y1 - y0)
    top, bot = max(0.0, y0 - pad), min(float(height), y1 + pad)
    h = bot - top
    w = max(0.62 * h, np.ptp(lm[:, 0]) * 1.4)
    cx = float(np.median(lm[:, 0]))
    left = min(max(0.0, cx - w / 2), max(0.0, width - w))
    return left, top, min(w, width), h


def measure_zoomed(predictor: Predictor, img: Image.Image, passes: int = 3) -> dict:
    """Run, crop to the spine the model found, run again; repeat until the box settles.

    Photos of a whole-body film show the spine at about half the height, where the
    vertebrae are too small for the model. Zooming in to the spine fixes that.
    """
    box = (0.0, 0.0, float(img.width), float(img.height))
    r = predictor(img)
    for _ in range(passes - 1):
        lm = np.array(r["landmarks"]) + [box[0], box[1]]
        nb = spine_box(lm, img.width, img.height)
        if max(abs(a - b) for a, b in zip(nb, box)) < 0.02 * img.height:
            break
        box = nb
        crop = img.crop((round(box[0]), round(box[1]), round(box[0] + box[2]), round(box[1] + box[3])))
        r = predictor(crop)
    lm = np.array(r["landmarks"]) + [box[0], box[1]]
    r["landmarks"] = np.round(lm, 1).tolist()
    r["curves"] = [c.to_dict() for c in curves_from_landmarks(lm)]
    r["crop_box"] = [round(v, 1) for v in box]
    return r

