"""Curve pattern classification.

Two parts:

1. Ground truth from landmarks (vertebra levels known): each curve's apex vertebra
   gives its region (SRS definitions: thoracic apex T2 to the T11-12 disc,
   thoracolumbar T12-L1, lumbar the L1-2 disc to L4), and the apex's side gives
   the direction it bends on the image.

2. A model for manual mode, where vertebra levels are NOT known: from the four
   marked endplate points and their position in the cropped image, predict the
   region. Softmax regression in numpy, exported to JSON for the browser.

    python pattern.py --synthetic 3000 --out pattern_model.json     # train + evaluate
    python pattern.py --data /path/to/boostnet_labeldata --out ...   # AASCE, once available
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from cobb import N_VERTEBRAE, VERTEBRA_NAMES, curves_from_landmarks, tilt_deg, angle_between

REGIONS = ["thoracic", "thoracolumbar", "lumbar"]
T12, L1 = 11, 12  # vertebra indices


def region_of_apex(apex: int) -> str:
    if apex < T12:
        return "thoracic"
    if apex <= L1:
        return "thoracolumbar"
    return "lumbar"


def centres(landmarks: np.ndarray) -> np.ndarray:
    return np.asarray(landmarks, float).reshape(N_VERTEBRAE, 4, 2).mean(axis=1)


def apex_and_side(landmarks: np.ndarray, upper_endplate: int, lower_endplate: int) -> tuple[int, str]:
    """Apex = vertebra between the end vertebrae furthest from the line joining their centres."""
    c = centres(landmarks)
    a, b = upper_endplate // 2, lower_endplate // 2
    p, q = c[a], c[b]
    d = q - p
    n = np.array([d[1], -d[0]]) / (np.linalg.norm(d) + 1e-9)   # points to image-right when going down
    ks = list(range(a, b + 1))
    dev = [(c[k] - p) @ n for k in ks]
    i = int(np.argmax(np.abs(dev)))
    side = "image-right" if dev[i] > 0 else "image-left"
    return ks[i], side


@dataclass
class CurveInfo:
    name: str
    cobb_deg: float
    region: str
    side: str
    apex: str

    def to_dict(self):
        return self.__dict__ | {"cobb_deg": round(self.cobb_deg, 1)}


MAJOR_FRACTION = 0.6  # a curve counts toward the pattern if it is at least 60% of the largest


def is_major(angle: float, largest: float, min_deg: float = 10.0) -> bool:
    """Small compensatory curves above and below a curve are always present geometrically;
    only curves comparable to the largest decide the pattern. Same rule as the browser."""
    return angle >= min_deg and angle >= MAJOR_FRACTION * largest


def classify_pattern(landmarks: np.ndarray, min_deg: float = 10.0) -> dict:
    """Pattern from landmarks: per-curve region, side and major/minor, plus an overall label."""
    curves = [c for c in curves_from_landmarks(landmarks) if c.cobb_deg >= min_deg]
    largest = max((c.cobb_deg for c in curves), default=0.0)
    infos, majors = [], []
    for c in curves:
        apex, side = apex_and_side(landmarks, c.upper_endplate, c.lower_endplate)
        info = CurveInfo(c.name, c.cobb_deg, region_of_apex(apex), side, VERTEBRA_NAMES[apex])
        major = is_major(c.cobb_deg, largest, min_deg)
        infos.append(info.to_dict() | {"major": major})
        if major:
            majors.append(info)
    return {"pattern": overall_pattern([i.region for i in majors], [i.cobb_deg for i in majors]),
            "curves": infos}


def overall_pattern(regions: list[str], angles: list[float]) -> str:
    if not regions:
        return "none"
    if len(regions) == 1:
        return f"single {regions[0]}"
    has_t = "thoracic" in regions
    has_l = any(r != "thoracic" for r in regions)
    if has_t and has_l:
        return "double (thoracic and thoracolumbar/lumbar)"
    return f"double {regions[int(np.argmax(angles))]}"


# ---------------------------------------------------------------- manual-mode features

def marks_from_landmarks(landmarks: np.ndarray, upper_endplate: int, lower_endplate: int) -> np.ndarray:
    """The 4 points a user would tap: top edge of the upper end vertebra, bottom edge of the lower one."""
    v = np.asarray(landmarks, float).reshape(N_VERTEBRAE, 4, 2)
    up, lo = v[upper_endplate // 2], v[lower_endplate // 2]
    return np.array([up[0], up[1], lo[2], lo[3]])


def features(marks: np.ndarray, image_h: float, image_w: float) -> np.ndarray:
    """Features available in manual mode. Keep in sync with patternFeatures() in web/index.html."""
    m = np.asarray(marks, float)
    y = m[:, 1] / image_h
    t1, t2 = tilt_deg(m[0], m[1]), tilt_deg(m[2], m[3])
    span = y[2:].mean() - y[:2].mean()
    mid = y.mean()
    xmid = m[:, 0].mean() / image_w
    return np.array([mid, y[:2].mean(), y[2:].mean(), span, mid ** 2,
                     angle_between(t1, t2) / 90, xmid])


class SoftmaxRegression:
    def __init__(self, n_classes: int = 3, lr: float = 0.5, epochs: int = 2000, l2: float = 1e-3):
        self.k, self.lr, self.epochs, self.l2 = n_classes, lr, epochs, l2

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Z = (X - self.mu) / self.sd
        n, d = Z.shape
        self.W, self.b = np.zeros((d, self.k)), np.zeros(self.k)
        Y = np.eye(self.k)[y]
        for _ in range(self.epochs):
            P = self._softmax(Z @ self.W + self.b)
            G = (P - Y) / n
            self.W -= self.lr * (Z.T @ G + self.l2 * self.W)
            self.b -= self.lr * G.sum(0)
        return self

    @staticmethod
    def _softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def predict_proba(self, X):
        return self._softmax(((X - self.mu) / self.sd) @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def to_json(self) -> dict:
        return {"classes": REGIONS, "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "W": self.W.tolist(), "b": self.b.tolist()}


def threshold_baseline(X: np.ndarray) -> np.ndarray:
    """The hand-written rule the browser used before a model existed (curve midpoint height)."""
    mid = X[:, 0]
    return np.where(mid < 0.58, 0, np.where(mid < 0.66, 1, 2))


# ---------------------------------------------------------------- synthetic spines

def synthetic_spine(segments, gap=57.0, cx=300.0, noise_px=0.0, rng=None):
    """Landmarks for a spine made of curve segments [(top_k, bottom_k, cobb_deg, sign), ...].

    Each segment is a half sine between its end vertebrae; vertebrae outside every
    segment are straight. A shared end vertebra takes the upper segment's tilt.
    """
    rng = rng or np.random.default_rng()
    pts = []
    for k in range(N_VERTEBRAE):
        yc = 100 + k * gap
        xc, m = cx, 0.0
        for top, bot, cobb, sign in segments:
            if top <= k <= bot:
                L = (bot - top) * gap
                A = np.tan(np.radians(cobb / 2)) * L / np.pi * sign
                t = (yc - (100 + top * gap)) / L
                xc = cx + A * np.sin(np.pi * t)
                m = A * np.pi / L * np.cos(np.pi * t)
                break
        n = np.hypot(m, 1)
        spine, plate = np.array([m / n, 1 / n]), np.array([1 / n, -m / n])
        w, h = 58 + 1.8 * k, 42
        c = np.array([xc, yc])
        for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            pts.append(c + sx * w / 2 * plate + sy * h / 2 * spine)
    lm = np.array(pts)
    return lm + rng.normal(0, noise_px, lm.shape) if noise_px else lm


def random_segments(rng):
    """One curve, or two curves sharing an end vertebra (S-shape)."""
    if rng.random() < 0.5:
        a = int(rng.integers(0, 12)); b = int(min(16, a + rng.integers(4, 9)))
        return [(a, b, float(rng.uniform(12, 70)), float(rng.choice([-1, 1])))]
    a = int(rng.integers(0, 5)); m = int(rng.integers(a + 4, min(a + 9, 13))); b = int(min(16, m + rng.integers(4, 8)))
    s = float(rng.choice([-1, 1]))
    return [(a, m, float(rng.uniform(15, 70)), s), (m, b, float(rng.uniform(12, 60)), -s)]


def synthetic_dataset(n: int, seed: int = 0):
    """X features from simulated taps, y true region from the segment's apex vertebra."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        segs = random_segments(rng)
        lm = synthetic_spine(segs, noise_px=0.8, rng=rng)
        # image frame like an auto-cropped photo: some neck/skull above T1, pelvis below L5
        top = lm[:, 1].min() - rng.uniform(0.05, 0.3) * 960
        bot = lm[:, 1].max() + rng.uniform(0.1, 0.3) * 960
        H, W = bot - top, 600 * rng.uniform(0.8, 1.2)
        lm_img = lm - np.array([0, top])
        for top_k, bot_k, _, _ in segs:
            marks = marks_from_landmarks(lm_img, 2 * top_k, 2 * bot_k + 1)
            marks = marks + rng.normal(0, 4, marks.shape)          # tapping error
            apex, _ = apex_and_side(lm_img, 2 * top_k, 2 * bot_k + 1)
            X.append(features(marks, H, W))
            y.append(REGIONS.index(region_of_apex(apex)))
    return np.array(X), np.array(y)


def aasce_dataset(root: str):
    from dataset import read_aasce
    X, y = [], []
    for s in read_aasce(root, "training") + read_aasce(root, "test"):
        for c in curves_from_landmarks(s.landmarks):
            if c.cobb_deg < 10:
                continue
            apex, _ = apex_and_side(s.landmarks, c.upper_endplate, c.lower_endplate)
            X.append(features(marks_from_landmarks(s.landmarks, c.upper_endplate, c.lower_endplate),
                              s.image.height, s.image.width))
            y.append(REGIONS.index(region_of_apex(apex)))
    return np.array(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--synthetic", type=int, default=3000)
    ap.add_argument("--out", default="pattern_model.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X, y = aasce_dataset(args.data) if args.data else synthetic_dataset(args.synthetic, args.seed)
    idx = np.random.default_rng(args.seed).permutation(len(X))
    cut = int(0.8 * len(X))
    tr, te = idx[:cut], idx[cut:]
    model = SoftmaxRegression().fit(X[tr], y[tr])
    acc = (model.predict(X[te]) == y[te]).mean()
    base = (threshold_baseline(X[te]) == y[te]).mean()
    print(f"curves {len(X)}  class counts {np.bincount(y, minlength=3).tolist()}")
    print(f"test accuracy  model {acc * 100:.1f}%   threshold rule {base * 100:.1f}%")
    conf = np.zeros((3, 3), int)
    for t, p in zip(y[te], model.predict(X[te])):
        conf[t, p] += 1
    print("confusion (rows true, cols predicted):", REGIONS)
    print(conf)
    out = model.to_json() | {"source": "aasce" if args.data else "synthetic", "test_accuracy": round(float(acc), 3)}
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"model -> {args.out}")


if __name__ == "__main__":
    main()
