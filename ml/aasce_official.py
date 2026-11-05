"""The AASCE 2019 challenge's own Cobb angle formula, for fair benchmarking.

The challenge labels (angles.csv) were made with this function, which measures
each vertebra's tilt along its midline (the line joining the midpoints of its
left and right sides) rather than along its endplates. SpineAngle measures
endplates in the app, as clinicians do; this module is only used so the
accuracy numbers in evaluate.py are comparable with published AASCE results.

Ported from the challenge's MATLAB reference (also used in open-source AASCE
solutions). Returns the three angles in the same order as angles.csv.
"""
from __future__ import annotations

import numpy as np


def _is_s(mid_p_v: np.ndarray) -> bool:
    num = mid_p_v.shape[0]
    ll = []
    for i in range(num - 2):
        t1 = (mid_p_v[i, 1] - mid_p_v[num - 1, 1]) / (mid_p_v[0, 1] - mid_p_v[num - 1, 1])
        t2 = (mid_p_v[i, 0] - mid_p_v[num - 1, 0]) / (mid_p_v[0, 0] - mid_p_v[num - 1, 0])
        ll.append(t1 - t2)
    ll = np.asarray(ll, np.float64)[:, None]
    pair = ll @ ll.T
    return abs(pair.sum() - np.abs(pair).sum()) >= 1e-4


def official_angles(pts: np.ndarray, image_height: float) -> np.ndarray:
    pts = np.asarray(pts, np.float64)
    n = pts.shape[0]
    vnum = n // 4 - 1
    mid_p_v = (pts[0::2] + pts[1::2]) / 2                      # endplate midpoints, 34 x 2
    mid_p = []
    for i in range(0, n, 4):
        mid_p.append((pts[i] + pts[i + 2]) / 2)                # left side midpoint
        mid_p.append((pts[i + 1] + pts[i + 3]) / 2)            # right side midpoint
    mid_p = np.asarray(mid_p)
    vec = mid_p[1::2] - mid_p[0::2]                            # vertebra midlines, 17 x 2
    mod = np.sqrt((vec ** 2).sum(1))[:, None]
    ang = np.arccos(np.clip((vec @ vec.T) / (mod @ mod.T), 0.0, 1.0))
    pos1 = ang.argmax(1)
    maxt = ang.max(1)
    pos2 = int(maxt.argmax())
    a1 = maxt.max()
    if not _is_s(mid_p_v):
        a2 = ang[0, pos2]
        a3 = ang[vnum, pos1[pos2]]
    elif mid_p_v[pos2 * 2, 1] + mid_p_v[pos1[pos2] * 2, 1] < image_height:
        a2 = ang[pos2, :pos2 + 1].max()
        a3 = ang[pos1[pos2], pos1[pos2]:vnum + 1].max()
    else:
        row = ang[pos2, :pos2 + 1]
        a2 = row.max()
        p11 = int(row.argmax())
        a3 = ang[p11, :p11 + 1].max()
    return np.degrees([a1, a2, a3])
