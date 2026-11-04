"""Data loading for AASCE-format spinal X-rays, plus a synthetic set for smoke runs.

Expected AASCE layout (check against your download, the README explains):

    <root>/data/{training,test}/*.jpg
    <root>/labels/{training,test}/filenames.csv   one image name per row
    <root>/labels/{training,test}/landmarks.csv   136 values per row:
                                                  68 x then 68 y, normalised to [0, 1]
    <root>/labels/{training,test}/angles.csv      3 Cobb angles per row
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from cobb import N_LANDMARKS, aasce_angles
from synthetic import synthetic_landmarks

INPUT_H, INPUT_W = 1024, 512   # model input, portrait like a full-spine film
STRIDE = 4                     # heatmaps are 1/4 resolution
SIGMA = 2.0                    # Gaussian sigma at heatmap resolution


@dataclass
class Letterbox:
    """Maps between original image pixels and model-input pixels."""
    scale: float
    pad_x: float
    pad_y: float

    def to_input(self, pts: np.ndarray) -> np.ndarray:
        return pts * self.scale + np.array([self.pad_x, self.pad_y])

    def to_original(self, pts: np.ndarray) -> np.ndarray:
        return (pts - np.array([self.pad_x, self.pad_y])) / self.scale


def letterbox(img: Image.Image, h: int = INPUT_H, w: int = INPUT_W) -> tuple[Image.Image, Letterbox]:
    """Resize keeping aspect ratio, then pad to h x w. Returns greyscale image."""
    img = img.convert("L")
    s = min(h / img.height, w / img.width)
    nw, nh = round(img.width * s), round(img.height * s)
    canvas = Image.new("L", (w, h), 0)
    px, py = (w - nw) // 2, (h - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    return canvas, Letterbox(s, px, py)


def to_array(img: Image.Image) -> np.ndarray:
    """Greyscale image to float32 1 x H x W, with histogram equalisation."""
    img = ImageOps.equalize(img)
    return (np.asarray(img, np.float32) / 255.0)[None]


def gaussian_heatmaps(pts_input: np.ndarray, h: int = INPUT_H // STRIDE,
                      w: int = INPUT_W // STRIDE, sigma: float = SIGMA) -> np.ndarray:
    """68 x h x w heatmaps with a Gaussian at each landmark."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.zeros((len(pts_input), h, w), np.float32)
    for k, (x, y) in enumerate(pts_input / STRIDE):
        out[k] = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
    return out


def random_affine(img: Image.Image, pts: np.ndarray, rng: np.random.Generator,
                  max_rot: float = 10, scale=(0.9, 1.1)) -> tuple[Image.Image, np.ndarray]:
    """Rotate and scale about the image centre. No flips: they would swap curve direction."""
    ang = math.radians(rng.uniform(-max_rot, max_rot))
    s = rng.uniform(*scale)
    cx, cy = img.width / 2, img.height / 2
    cos, sin = math.cos(ang) * s, math.sin(ang) * s
    # forward map for points
    fwd = np.array([[cos, -sin], [sin, cos]])
    new_pts = (pts - [cx, cy]) @ fwd.T + [cx, cy]
    # PIL wants the inverse map (output -> input)
    inv = np.linalg.inv(fwd)
    a, b, d, e = inv[0, 0], inv[0, 1], inv[1, 0], inv[1, 1]
    c = cx - a * cx - b * cy
    f = cy - d * cx - e * cy
    out = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f), resample=Image.BILINEAR)
    return out, new_pts


def jitter(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(img, np.float32)
    arr = arr * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


@dataclass
class Sample:
    image: Image.Image          # original image
    landmarks: np.ndarray       # 68 x 2 in original pixels
    angles: np.ndarray          # 3 ground-truth Cobb angles
    name: str


def read_aasce(root: str | Path, split: str) -> list[Sample]:
    """Read one AASCE split ('training', 'test')."""
    root = Path(root)
    lab = root / "labels" / split

    def rows(name):
        with open(lab / name, newline="") as fh:
            return [r for r in csv.reader(fh) if r]

    names = [r[0] for r in rows("filenames.csv")]
    lms = np.array(rows("landmarks.csv"), float)
    angs = np.array(rows("angles.csv"), float)
    if not (len(names) == len(lms) == len(angs)):
        raise ValueError(f"{split}: {len(names)} names, {len(lms)} landmark rows, {len(angs)} angle rows")
    out = []
    for name, lm, ang in zip(names, lms, angs):
        img = Image.open(root / "data" / split / name)
        pts = np.stack([lm[:N_LANDMARKS] * img.width, lm[N_LANDMARKS:] * img.height], axis=1)
        out.append(Sample(img, pts, ang, name))
    return out


def render_synthetic(lm: np.ndarray, size=(600, 1150), seed: int = 0) -> Image.Image:
    """Draw vertebral bodies as bright quadrilaterals on a dark radial background."""
    rng = np.random.default_rng(seed)
    w, h = size
    ys, xs = np.mgrid[0:h, 0:w]
    r = np.hypot((xs - w / 2) / w, (ys - h * 0.45) / h)
    bg = (60 * np.clip(1 - 1.6 * r, 0, 1) + rng.normal(0, 6, (h, w))).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(bg)
    d = ImageDraw.Draw(img)
    for v in lm.reshape(-1, 4, 2):
        tl, tr, bl, br = v
        d.polygon([tuple(tl), tuple(tr), tuple(br), tuple(bl)], fill=int(rng.uniform(170, 215)))
    return img.filter(ImageFilter.GaussianBlur(1.5))


def make_synthetic(n: int, seed: int = 0) -> list[Sample]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        lm = synthetic_landmarks(rng.uniform(5, 70), top=int(rng.integers(2, 7)),
                                 bottom=int(rng.integers(11, 16)), noise_px=0.5, seed=seed + i)
        lm[:, 0] += rng.uniform(-40, 40)
        out.append(Sample(render_synthetic(lm, seed=seed + i), lm, aasce_angles(lm), f"synthetic_{i:04d}"))
    return out


class SpineDataset:
    """PyTorch-style dataset: returns (image 1xHxW, heatmaps 68xhxw, landmarks in input px)."""

    def __init__(self, samples: list[Sample], augment: bool = False, seed: int = 0):
        self.samples, self.augment = samples, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img, pts = s.image.convert("L"), s.landmarks.copy()
        if self.augment:
            img, pts = random_affine(img, pts, self.rng)
            img = jitter(img, self.rng)
        boxed, lb = letterbox(img)
        pts_in = lb.to_input(pts).astype(np.float32)
        return to_array(boxed), gaussian_heatmaps(pts_in), pts_in
