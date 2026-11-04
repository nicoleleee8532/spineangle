"""Synthetic spines with a known Cobb angle, for tests and pipeline smoke runs.

Same construction as the example in web/index.html: vertebra centres follow a
sine curve whose slope at the two end vertebrae gives the requested angle.
"""
from __future__ import annotations

import numpy as np

from cobb import N_VERTEBRAE


def synthetic_landmarks(cobb_deg: float, top: int = 4, bottom: int = 13,
                        gap: float = 57.0, cx: float = 300.0,
                        noise_px: float = 0.0, seed: int | None = None) -> np.ndarray:
    """68 x 2 landmarks for a single C-curve between vertebrae ``top`` and ``bottom``.

    Each vertebra's endplates are perpendicular to the spine's tangent at its
    centre, so the end vertebrae tilt by +/- cobb_deg / 2.
    """
    rng = np.random.default_rng(seed)
    y0 = 100 + top * gap
    L = (bottom - top) * gap
    s = np.tan(np.radians(cobb_deg / 2))
    A = s * L / np.pi
    pts = []
    for k in range(N_VERTEBRAE):
        yc = y0 + (k - top) * gap
        xc = cx + A * np.sin(np.pi * (yc - y0) / L)
        m = A * np.pi / L * np.cos(np.pi * (yc - y0) / L)
        n = np.hypot(m, 1)
        spine = np.array([m / n, 1 / n])
        plate = np.array([1 / n, -m / n])
        w, h = 58 + 1.8 * k, 42
        c = np.array([xc, yc])
        for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:  # TL, TR, BL, BR
            pts.append(c + sx * w / 2 * plate + sy * h / 2 * spine)
    lm = np.array(pts)
    if noise_px:
        lm = lm + rng.normal(0, noise_px, lm.shape)
    return lm
