"""Export a trained checkpoint to ONNX for the browser (onnxruntime-web).

    python export_onnx.py checkpoints/best.pt ../web/model/spine-landmarks.onnx

The exported graph takes a 1 x 1 x H x W float image (letterboxed, histogram
equalised, scaled to [0, 1] exactly as dataset.to_array does) and returns
68 x 2 landmark coordinates in input pixels plus confidence scores, so the
page does no heatmap maths itself. Weights are stored as float16 to halve the
download; the maths still runs in float32.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from dataset import INPUT_H, INPUT_W, STRIDE
from model import LandmarkUNet, soft_argmax
from vertebra import VertebraNet, decode


class Exported(torch.nn.Module):
    """Wraps either model so both export the same outputs: 68 x 2 landmarks, scores."""

    def __init__(self, net, kind):
        super().__init__()
        self.net, self.kind = net, kind

    def forward(self, x):
        if self.kind == "vertebra":
            lm, score = decode(self.net(x))         # score per vertebra (17)
            return lm[0], score[0]
        coords, peak = soft_argmax(self.net(x))
        return coords[0] * STRIDE, peak[0]


def to_fp16_weights(path: Path) -> None:
    """Store big float32 initialisers as float16 + a Cast node back to float32."""
    m = onnx.load(str(path))
    g = m.graph
    new_inits, casts = [], []
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32 and arr.size >= 1024:
            half = numpy_helper.from_array(arr.astype(np.float16), init.name + "_fp16")
            new_inits.append(half)
            casts.append(onnx.helper.make_node("Cast", [half.name], [init.name], to=onnx.TensorProto.FLOAT))
        else:
            new_inits.append(init)
    del g.initializer[:]
    g.initializer.extend(new_inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def main(ckpt_path: str, out_path: str) -> None:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # our own checkpoint
    kind = ck.get("kind", "heatmap")
    net = VertebraNet(base=ck.get("base", 16)) if kind == "vertebra" else LandmarkUNet(base=ck.get("base", 32))
    net.load_state_dict(ck["model"])
    model = Exported(net.eval(), kind).eval()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, 1, INPUT_H, INPUT_W)
    torch.onnx.export(model, (x,), str(out), input_names=["image"], output_names=["landmarks", "confidence"],
                      opset_version=17, dynamo=False)
    to_fp16_weights(out)

    import onnxruntime as ort
    ref = [t.detach().numpy() for t in model(torch.rand(1, 1, INPUT_H, INPUT_W, generator=torch.Generator().manual_seed(0)))]
    sess = ort.InferenceSession(str(out))
    got = sess.run(None, {"image": torch.rand(1, 1, INPUT_H, INPUT_W, generator=torch.Generator().manual_seed(0)).numpy()})
    err = float(np.abs(ref[0] - got[0]).max())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), max landmark difference vs PyTorch {err:.2f}px")


if __name__ == "__main__":
    main(*sys.argv[1:3])
