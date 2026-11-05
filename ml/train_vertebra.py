"""Train the vertebra-centre model (vertebra.py) on AASCE.

    python train_vertebra.py --data /path/to/boostnet_labeldata --epochs 60 --out runs/vert
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cobb import curves_from_landmarks
from dataset import INPUT_H, INPUT_W, SpineDataset, read_aasce, shrink
from train import split
from vertebra import STRIDE, VertebraNet, decode, loss_fn, targets


class Wrap(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, _, pts = self.ds[i]
        hm, ind, reg, valid = targets(pts, INPUT_H // STRIDE, INPUT_W // STRIDE)
        return [torch.from_numpy(a) for a in (x, hm, ind, reg, valid, pts)]


@torch.no_grad()
def validate(model, loader):
    model.eval()
    tot, n, lm_err, cobb_err = np.zeros(4), 0, [], []
    for x, hm, ind, reg, valid, pts in loader:
        out = model(x)
        loss, parts = loss_fn(out, hm, ind, reg, valid)
        tot += np.array([loss.item(), *parts]) * x.shape[0]
        n += x.shape[0]
        lm, _ = decode(out)
        for p, t in zip(lm.numpy(), pts.numpy()):
            lm_err.append(np.linalg.norm(p - t, axis=1).mean())
            cobb_err.append(abs(curves_from_landmarks(p)[0].cobb_deg - curves_from_landmarks(t)[0].cobb_deg))
    r = {k: float(v) for k, v in zip(["loss", "hm", "off", "vec"], tot / n)}
    r["landmark_px"], r["cobb_mae"] = float(np.mean(lm_err)), float(np.mean(cobb_err))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.25e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--resume")
    ap.add_argument("--init", help="start from these weights (fresh optimiser and schedule)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)

    tr, va = split(shrink(read_aasce(args.data, "training")), args.val_frac, args.seed)
    print(f"train {len(tr)}  val {len(va)}", flush=True)
    dl_tr = DataLoader(Wrap(SpineDataset(tr, augment=True, seed=args.seed)), batch_size=args.batch,
                       shuffle=True, num_workers=args.workers, persistent_workers=args.workers > 0)
    dl_va = DataLoader(Wrap(SpineDataset(va)), batch_size=args.batch, num_workers=0)

    model = VertebraNet(base=args.base)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl_tr),
                                                pct_start=0.1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["model"])
        print(f"initialised from {args.init}", flush=True)
    best, start = float("inf"), 1
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {ck['epoch']}", flush=True)

    for ep in range(start, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for x, hm, ind, reg, valid, pts in dl_tr:
            loss, _ = loss_fn(model(x), hm, ind, reg, valid)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * x.shape[0]
            n += x.shape[0]
        m = validate(model, dl_va)
        print(f"ep {ep:3d}  train {tot / n:.3f}  val {m['loss']:.3f} (hm {m['hm']:.3f} off {m['off']:.3f} "
              f"vec {m['vec']:.3f})  landmark {m['landmark_px']:.1f}px  Cobb MAE {m['cobb_mae']:.2f}°  "
              f"({time.time() - t0:.0f}s)", flush=True)
        ck = {"kind": "vertebra", "model": model.state_dict(), "base": args.base, "epoch": ep,
              "version": f"vertebra-b{args.base}-ep{ep}", "val": m}
        if m["cobb_mae"] < best:
            best = m["cobb_mae"]
            torch.save(ck, out / "best.pt")
        torch.save({**ck, "opt": opt.state_dict(), "sched": sched.state_dict(), "best": best}, out / "last.pt")
    print(f"best val Cobb MAE {best:.2f}°", flush=True)


if __name__ == "__main__":
    main()
