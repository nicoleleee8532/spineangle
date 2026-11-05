"""Train the landmark model.

Real data:   python train.py --data /path/to/boostnet_labeldata --epochs 150
Smoke test:  python train.py --synthetic 64 --epochs 2 --batch 4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataset import STRIDE, SpineDataset, make_synthetic, read_aasce, shrink
from model import LandmarkUNet, cobb_from_coords, soft_argmax


class TorchWrap(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, hm, pts = self.ds[i]
        return torch.from_numpy(x), torch.from_numpy(hm), torch.from_numpy(pts)


def split(samples, val_frac: float, seed: int):
    idx = np.random.default_rng(seed).permutation(len(samples))
    n_val = max(1, int(len(samples) * val_frac))
    return [samples[i] for i in idx[n_val:]], [samples[i] for i in idx[:n_val]]


def run_epoch(model, loader, opt, device, cobb_weight, train: bool):
    model.train(train)
    totals = {"loss": 0.0, "landmark_px": 0.0, "cobb_mae": 0.0}
    n = 0
    for x, hm, pts in loader:
        x, hm, pts = x.to(device), hm.to(device), pts.to(device)
        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = F.mse_loss(pred, hm) * 1000          # heatmap values are small; rescale
            coords, _ = soft_argmax(pred)
            coords = coords * STRIDE
            cobb_pred, cobb_true = cobb_from_coords(coords), cobb_from_coords(pts)
            if cobb_weight:
                loss = loss + cobb_weight * F.l1_loss(cobb_pred, cobb_true)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        b = x.shape[0]
        totals["loss"] += loss.item() * b
        totals["landmark_px"] += (coords - pts).norm(dim=-1).mean().item() * b
        totals["cobb_mae"] += (cobb_pred - cobb_true).abs().mean().item() * b
        n += b
    return {k: v / n for k, v in totals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="AASCE root folder")
    ap.add_argument("--synthetic", type=int, default=0, help="use N synthetic images instead")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32, help="U-Net width")
    ap.add_argument("--cobb-weight", type=float, default=0.0,
                    help="weight of the auxiliary Cobb L1 loss; try 0.1 after the heatmaps converge")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", help="checkpoint to continue from (keeps the epoch count and schedule)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.synthetic:
        samples = make_synthetic(args.synthetic, seed=args.seed)
    elif args.data:
        samples = shrink(read_aasce(args.data, "training"))
    else:
        ap.error("pass --data or --synthetic")
    tr, va = split(samples, args.val_frac, args.seed)
    print(f"train {len(tr)}  val {len(va)}")

    dl_tr = DataLoader(TorchWrap(SpineDataset(tr, augment=True, seed=args.seed)), batch_size=args.batch,
                       shuffle=True, num_workers=args.workers)
    dl_va = DataLoader(TorchWrap(SpineDataset(va)), batch_size=args.batch, num_workers=args.workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LandmarkUNet(base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    best, start = float("inf"), 1
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
        start, best = ck["epoch"] + 1, ck.get("best", best)
        print(f"resumed from epoch {ck['epoch']}")
    for ep in range(start, args.epochs + 1):
        t0 = time.time()
        m_tr = run_epoch(model, dl_tr, opt, device, args.cobb_weight, train=True)
        m_va = run_epoch(model, dl_va, opt, device, args.cobb_weight, train=False)
        sched.step()
        print(f"ep {ep:3d}  train loss {m_tr['loss']:.3f}  val loss {m_va['loss']:.3f}  "
              f"val landmark {m_va['landmark_px']:.1f}px  val Cobb MAE {m_va['cobb_mae']:.2f}°  "
              f"({time.time() - t0:.0f}s)")
        ckpt = {"model": model.state_dict(), "base": args.base, "epoch": ep,
                "version": f"unet-b{args.base}-ep{ep}", "val": m_va}
        if m_va["cobb_mae"] < best:
            best = m_va["cobb_mae"]
            torch.save(ckpt, out / "best.pt")
        torch.save({**ckpt, "opt": opt.state_dict(), "sched": sched.state_dict(), "best": best}, out / "last.pt")
    print(f"best val Cobb MAE {best:.2f}°  ->  {out / 'best.pt'}")


if __name__ == "__main__":
    main()
