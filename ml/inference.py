"""Run a trained model on one image and return landmarks, confidences and curves."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cobb import curves_from_landmarks
from dataset import STRIDE, letterbox, to_array
from model import LandmarkUNet, soft_argmax

LOW_CONF_MEAN = 0.3      # mean peak height below this -> ask the user to check
COLLAPSE_PX = 3.0        # all 4 corners of a vertebra within this many px -> suspicious


class Predictor:
    def __init__(self, weights: str | Path, device: str = "cpu"):
        ckpt = torch.load(weights, map_location=device)
        self.version = ckpt.get("version", "unet-v0.1")
        self.model = LandmarkUNet(base=ckpt.get("base", 32)).to(device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.device = device

    @torch.no_grad()
    def __call__(self, img: Image.Image) -> dict:
        boxed, lb = letterbox(img)
        x = torch.from_numpy(to_array(boxed))[None].to(self.device)
        hm = self.model(x)
        coords, peak = soft_argmax(hm)
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
