"""Evaluate a checkpoint on the test split.

    python evaluate.py --weights checkpoints/best.pt --data /path/to/boostnet_labeldata
    python evaluate.py --weights checkpoints/best.pt --synthetic 32

Writes results.csv (per image) and prints landmark error, Cobb MAE per curve,
SMAPE and severity-band accuracy. Add --plot for a Bland-Altman chart (needs matplotlib).

Two ways of scoring the angles:
  * challenge: the AASCE formula (aasce_official.py) on our landmarks vs the
    labels in angles.csv, so SMAPE is comparable with published results;
  * app: the endplate method the website uses (cobb.py), on our landmarks vs
    the same method on the doctors' landmarks.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from aasce_official import official_angles
from cobb import curves_from_landmarks, severity_band, smape
from dataset import make_synthetic, read_aasce
from inference import Predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    samples = make_synthetic(args.synthetic, seed=999) if args.synthetic else read_aasce(args.data, "test")
    pred_fn = Predictor(args.weights)

    rows, P, T, mre, app_p, app_t = [], [], [], [], [], []
    for s in samples:
        r = pred_fn(s.image)
        lm = np.array(r["landmarks"])
        p = official_angles(lm, s.image.height)
        P.append(p)
        T.append(s.angles)
        app_p.append(curves_from_landmarks(lm)[0].cobb_deg)
        app_t.append(curves_from_landmarks(s.landmarks)[0].cobb_deg)
        # landmark error as % of image height, since the films differ in size
        mre.append(np.linalg.norm(lm - s.landmarks, axis=1).mean() / s.image.height * 100)
        rows.append([s.name, *np.round(p, 1), *np.round(s.angles, 1), round(app_p[-1], 1), round(app_t[-1], 1),
                     round(mre[-1], 2), r["low_confidence"]])

    P, T, app_p, app_t = np.array(P), np.array(T), np.array(app_p), np.array(app_t)
    mae = np.abs(P - T).mean(axis=0)
    big = np.abs(P.max(1) - T.max(1))
    app = np.abs(app_p - app_t)
    band_acc = np.mean([severity_band(p) == severity_band(t) for p, t in zip(app_p, app_t)])
    print(f"images                     {len(samples)}")
    print(f"landmark error             {np.mean(mre):.2f} % of image height")
    print(f"challenge Cobb MAE (3 curves) {mae[0]:.2f}° / {mae[1]:.2f}° / {mae[2]:.2f}°")
    print(f"challenge SMAPE            {smape(P, T):.2f} %")
    print(f"largest curve MAE          {big.mean():.2f}°  (within 5°: {np.mean(big <= 5) * 100:.0f}%)")
    print(f"app method MAE             {app.mean():.2f}°  (within 5°: {np.mean(app <= 5) * 100:.0f}%, within 10°: {np.mean(app <= 10) * 100:.0f}%)")
    print(f"app severity band match    {band_acc * 100:.1f} %")

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "pred_pt", "pred_mt", "pred_tl", "true_pt", "true_mt", "true_tl",
                    "app_pred", "app_true", "landmark_err_pct_height", "low_confidence"])
        w.writerows(rows)
    print(f"per-image results -> {args.out}")

    if args.plot:
        import matplotlib.pyplot as plt
        p, t = P.max(1), T.max(1)
        mean, diff = (p + t) / 2, p - t
        bias, sd = diff.mean(), diff.std()
        plt.figure(figsize=(6, 4))
        plt.scatter(mean, diff, s=12)
        for y, ls in [(bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")]:
            plt.axhline(y, ls=ls, c="k", lw=1)
        plt.xlabel("Mean of model and ground truth (°)")
        plt.ylabel("Model − ground truth (°)")
        plt.title("Bland–Altman, largest curve")
        plt.tight_layout()
        plt.savefig("bland_altman.png", dpi=150)
        print("plot -> bland_altman.png")


if __name__ == "__main__":
    main()
