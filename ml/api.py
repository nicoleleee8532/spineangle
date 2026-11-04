"""Inference API. Images are processed in memory and never written to disk or logged.

    SPINEANGLE_WEIGHTS=checkpoints/best.pt uvicorn api:app --port 8000
"""
from __future__ import annotations

import io
import logging
import os
import time

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024
MIN_SIDE = 256

log = logging.getLogger("spineangle")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="SpineAngle", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("SPINEANGLE_ORIGINS", "*").split(","),
                   allow_methods=["POST", "GET"], allow_headers=["*"])

_predictor = None


def predictor():
    global _predictor
    if _predictor is None:
        weights = os.getenv("SPINEANGLE_WEIGHTS", "checkpoints/best.pt")
        if os.path.exists(weights):
            from inference import Predictor
            _predictor = Predictor(weights)
    return _predictor


def err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, "message": message})


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_loaded": predictor() is not None}


@app.post("/v1/analyse")
async def analyse(image: UploadFile = File(...)):
    t0 = time.perf_counter()
    data = await image.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        return err(413, "too_large", "Images must be 10 MB or smaller. Export a smaller JPEG.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        return err(400, "not_an_image", "Couldn't read that file. Upload a PNG or JPEG X-ray.")
    if min(img.size) < MIN_SIDE:
        return err(422, "too_small", f"The image is under {MIN_SIDE} px on one side. Upload a higher-resolution X-ray.")
    if img.height < img.width:
        return err(422, "wrong_view", "This looks like a landscape image. Upload a full-length frontal spine X-ray.")
    p = predictor()
    if p is None:
        return err(503, "model_not_loaded", "The model isn't available right now. Use manual mode instead.")

    result = p(img)
    # log only metadata, never pixels or landmarks
    log.info("analyse ok ms=%d low_conf=%s", (time.perf_counter() - t0) * 1000, result["low_confidence"])
    return result
