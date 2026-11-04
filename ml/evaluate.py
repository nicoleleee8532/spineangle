"""Evaluate a checkpoint on the test split.

    python evaluate.py --weights checkpoints/best.pt --data /path/to/boostnet_labeldata
    python evaluate.py --weights checkpoints/best.pt --synthetic 32

Writes results.csv (per image) and prints MRE, Cobb MAE per curve, SMAPE and
severity-band accuracy. Add --plot for a Bland-Altman chart (needs matplotlib).
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from cobb import aasce_angles, severity_band, smape
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

    rows, P, T, mre = [], [], [], []
    for s in samples:
        r = pred_fn(s.image)
        lm = np.array(r["landmarks"])
        p = aasce_angles(lm)
        P.append(p)
        T.append(s.angles)
        mre.append(np.linalg.norm(lm - s.landmarks, axis=1).mean())
        rows.append([s.name, *np.round(p, 1), *np.round(s.angles, 1), round(mre[-1], 1), r["low_confidence"]])

    P, T = np.array(P), np.array(T)
    mae = np.abs(P - T).mean(axis=0)
    band_acc = np.mean([severity_band(p) == severity_band(t) for p, t in zip(P.max(1), T.max(1))])
    print(f"images            {len(samples)}")
    print(f"landmark MRE      {np.mean(mre):.1f} px")
    print(f"Cobb MAE (PT/MT/TL) {mae[0]:.2f}° / {mae[1]:.2f}° / {mae[2]:.2f}°")
    print(f"SMAPE             {smape(P, T):.2f} %")
    print(f"band accuracy     {band_acc * 100:.1f} %")

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "pred_pt", "pred_mt", "pred_tl", "true_pt", "true_mt", "true_tl",
                    "mre_px", "low_confidence"])
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
