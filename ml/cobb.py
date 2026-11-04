"""Cobb angle geometry from vertebral landmarks.

Pure numpy, no torch, so it can be unit-tested anywhere and shared by the
training, evaluation and API code. The maths matches web/index.html.

Landmark layout (AASCE 2019): 17 vertebrae from T1 to L5, 4 corners each,
68 points in order top-left, top-right, bottom-left, bottom-right.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_VERTEBRAE = 17
N_LANDMARKS = N_VERTEBRAE * 4
VERTEBRA_NAMES = [f"T{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)]


def tilt_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Tilt of the line a -> b in degrees, normalised to (-90, 90]."""
    t = float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))
    while t > 90:
        t -= 180
    while t <= -90:
        t += 180
    return t


def angle_between(t1: float, t2: float) -> float:
    """Acute angle between two lines given their tilts in degrees."""
    d = abs(t1 - t2)
    return 180 - d if d > 90 else d


def manual_cobb(upper_a, upper_b, lower_a, lower_b) -> float:
    """Cobb angle from two marked endplates (the manual mode in the web tool)."""
    return angle_between(
        tilt_deg(np.asarray(upper_a, float), np.asarray(upper_b, float)),
        tilt_deg(np.asarray(lower_a, float), np.asarray(lower_b, float)),
    )


def endplate_tilts(landmarks: np.ndarray) -> np.ndarray:
    """34 endplate tilts, top to bottom: T1 upper, T1 lower, T2 upper, ... L5 lower."""
    lm = np.asarray(landmarks, float).reshape(N_VERTEBRAE, 4, 2)
    tilts = []
    for v in lm:
        tl, tr, bl, br = v
        tilts.append(tilt_deg(tl, tr))
        tilts.append(tilt_deg(bl, br))
    return np.array(tilts)


@dataclass
class Curve:
    name: str
    cobb_deg: float
    upper_endplate: int
    lower_endplate: int

    @property
    def upper_vertebra(self) -> str:
        return VERTEBRA_NAMES[self.upper_endplate // 2]

    @property
    def lower_vertebra(self) -> str:
        return VERTEBRA_NAMES[self.lower_endplate // 2]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "cobb_deg": round(self.cobb_deg, 1),
            "upper_endplate": self.upper_endplate,
            "lower_endplate": self.lower_endplate,
            "upper_vertebra": self.upper_vertebra,
            "lower_vertebra": self.lower_vertebra,
        }


def _max_pair(tilts: np.ndarray, lo: int, hi: int) -> tuple[float, int, int]:
    """Largest angle between any two endplates with indices in [lo, hi]."""
    best = (0.0, lo, lo)
    for i in range(lo, hi + 1):
        for j in range(i + 1, hi + 1):
            a = angle_between(tilts[i], tilts[j])
            if a > best[0]:
                best = (a, i, j)
    return best


def curves_from_landmarks(landmarks: np.ndarray, min_secondary_deg: float = 5.0) -> list[Curve]:
    """Primary curve plus up to two secondary curves (above and below it).

    The primary curve is the pair of endplates with the greatest angle between
    them. Secondary curves are the same search restricted to the endplates
    above the primary's upper end, and below its lower end. Secondary curves
    under ``min_secondary_deg`` are dropped as noise.
    """
    t = endplate_tilts(landmarks)
    n = len(t)
    a, i, j = _max_pair(t, 0, n - 1)
    curves = [Curve("primary", a, i, j)]
    if i >= 1:
        a2, i2, j2 = _max_pair(t, 0, i)
        if a2 >= min_secondary_deg:
            curves.append(Curve("upper", a2, i2, j2))
    if j <= n - 2:
        a3, i3, j3 = _max_pair(t, j, n - 1)
        if a3 >= min_secondary_deg:
            curves.append(Curve("lower", a3, i3, j3))
    return curves


def aasce_angles(landmarks: np.ndarray) -> np.ndarray:
    """Three angles in the AASCE order (proximal thoracic, main thoracic,
    thoracolumbar/lumbar), with 0 where a curve is absent.

    Simplification: primary goes in the middle slot, the upper secondary in
    the first and the lower secondary in the last.
    """
    out = np.zeros(3)
    for c in curves_from_landmarks(landmarks, min_secondary_deg=0.0):
        out[{"upper": 0, "primary": 1, "lower": 2}[c.name]] = c.cobb_deg
    return out


def smape(pred: np.ndarray, true: np.ndarray) -> float:
    """AASCE challenge metric, in percent. Arrays are (n_images, 3)."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    num = np.abs(pred - true).sum(axis=1)
    den = (pred + true).sum(axis=1)
    return float(np.mean(num / np.maximum(den, 1e-8)) * 100)


def severity_band(angle: float) -> str:
    if angle < 10:
        return "below scoliosis range"
    if angle < 25:
        return "mild"
    if angle < 40:
        return "moderate"
    return "severe"
