"""Auto-crop a photo of an X-ray (e.g. a phone photo of a hospital screen) to the skeleton.

Same steps as autoCropBox in web/index.html. Usage: python autocrop.py in.jpg out.jpg debug.jpg
"""
import sys
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

W = 240  # working width


def autocrop_box(img: Image.Image):
    s = W / img.width
    h = round(img.height * s)
    a = np.asarray(img.convert("RGB").resize((W, h))).astype(float)
    mx, mn = a.max(-1), a.min(-1)
    sat = (mx - mn) / (mx + 1)
    mask = (mx > 105) & (sat < 0.2)
    # remove flat UI: a pixel whose 9x9 neighbourhood barely varies is a toolbar or frame, not bone
    g = mx
    pad = np.pad(g, 4, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, (9, 9))
    flat = win.std(axis=(-1, -2)) < 6
    mask &= ~flat
    # remove long horizontal runs (toolbars, taskbar, screen edges): a body is never this wide in one run
    max_run = int(0.4 * W)
    for y in range(h):
        x = 0
        while x < W:
            if mask[y, x]:
                e = x
                while e < W and mask[y, e]:
                    e += 1
                if e - x > max_run:
                    mask[y, x:e] = False
                x = e
            else:
                x += 1

    def erode(m):
        o = m.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            o &= np.roll(np.roll(m, dy, 0), dx, 1)
        return o

    def dilate(m):
        o = m.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            o |= np.roll(np.roll(m, dy, 0), dx, 1)
        return o

    # opening removes 1-2 px lines (slanted screen edges); then join the blob
    mask = dilate(erode(mask))
    d = dilate(mask)
    # largest connected component, scored by original bright pixels
    lab = np.zeros(d.shape, int)
    best, best_score, cur = None, 0, 0
    for y in range(h):
        for x in range(W):
            if d[y, x] and not lab[y, x]:
                cur += 1
                q = deque([(y, x)]); lab[y, x] = cur
                ys, xs, score = [], [], 0
                while q:
                    cy, cx = q.popleft(); ys.append(cy); xs.append(cx); score += mask[cy, cx]
                    for ny, nx in ((cy+1, cx), (cy-1, cx), (cy, cx+1), (cy, cx-1)):
                        if 0 <= ny < h and 0 <= nx < W and d[ny, nx] and not lab[ny, nx]:
                            lab[ny, nx] = cur; q.append((ny, nx))
                # prefer big, roughly portrait blobs; penalise thin strips like toolbars
                bw, bh = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
                fill = score / (bw * bh)
                sc = score * min(1.0, bh / bw) if fill > 0.05 else 0
                if sc > best_score:
                    best_score, best = sc, (min(xs), min(ys), max(xs), max(ys))
    x0, y0, x1, y1 = best
    pad = 4
    return tuple(int(round(v / s)) for v in (max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(h, y1 + pad)))


if __name__ == "__main__":
    img = Image.open(sys.argv[1])
    box = autocrop_box(img)
    print("box", box)
    img.crop(box).save(sys.argv[2])
    dbg = img.convert("RGB").copy(); ImageDraw.Draw(dbg).rectangle(box, outline=(255, 0, 0), width=8)
    dbg.resize((dbg.width // 3, dbg.height // 3)).save(sys.argv[3])
