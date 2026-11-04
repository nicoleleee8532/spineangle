"""Checks that the torch, dataset and API code agree with cobb.py. Skipped without torch."""
import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch = pytest.importorskip("torch")

from cobb import curves_from_landmarks  # noqa: E402
from dataset import SpineDataset, letterbox, make_synthetic, random_affine  # noqa: E402
from model import LandmarkUNet, cobb_from_coords, soft_argmax  # noqa: E402
from synthetic import synthetic_landmarks  # noqa: E402


@pytest.mark.parametrize("angle", [0, 20, 45, 70])
def test_torch_cobb_matches_numpy(angle):
    lm = synthetic_landmarks(angle)
    t = cobb_from_coords(torch.tensor(lm[None], dtype=torch.float32)).item()
    assert t == pytest.approx(curves_from_landmarks(lm)[0].cobb_deg, abs=0.05)


def test_torch_cobb_ignores_corner_order_flip():
    lm = synthetic_landmarks(30).reshape(17, 4, 2)[:, [1, 0, 3, 2]].reshape(68, 2)  # swap left/right
    assert cobb_from_coords(torch.tensor(lm[None], dtype=torch.float32)).item() == pytest.approx(30, abs=0.05)


def test_affine_preserves_cobb():
    s = make_synthetic(1, seed=3)[0]
    _, pts = random_affine(s.image, s.landmarks, np.random.default_rng(0))
    assert curves_from_landmarks(pts)[0].cobb_deg == pytest.approx(curves_from_landmarks(s.landmarks)[0].cobb_deg, abs=0.01)


def test_letterbox_round_trip():
    s = make_synthetic(1)[0]
    _, lb = letterbox(s.image)
    assert np.allclose(lb.to_original(lb.to_input(s.landmarks)), s.landmarks)


def test_soft_argmax_finds_target_peaks():
    x, hm, pts = SpineDataset(make_synthetic(1))[0]
    coords, peak = soft_argmax(torch.from_numpy(hm)[None])
    assert (coords[0] * 4 - torch.from_numpy(pts)).norm(dim=-1).max() < 4
    assert peak.min() > 0.5


def test_model_output_shape():
    out = LandmarkUNet(base=4)(torch.zeros(1, 1, 1024, 512))
    assert out.shape == (1, 68, 256, 128)


def test_api_errors_without_model(monkeypatch):
    from fastapi.testclient import TestClient
    from PIL import Image
    monkeypatch.setenv("SPINEANGLE_WEIGHTS", "/nonexistent.pt")
    import api
    api._predictor = None
    c = TestClient(api.app)
    assert c.get("/healthz").json()["model_loaded"] is False
    assert c.post("/v1/analyse", files={"image": ("x.txt", b"hello", "text/plain")}).status_code == 400
    buf = io.BytesIO()
    Image.new("L", (600, 300)).save(buf, "PNG")
    assert c.post("/v1/analyse", files={"image": ("w.png", buf.getvalue(), "image/png")}).json()["error"] == "wrong_view"
    buf = io.BytesIO()
    Image.new("L", (600, 1200)).save(buf, "PNG")
    assert c.post("/v1/analyse", files={"image": ("ok.png", buf.getvalue(), "image/png")}).status_code == 503
