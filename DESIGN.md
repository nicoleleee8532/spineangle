# SpineAngle: System Design

Cobb angle measurement from spinal X-rays, with a manual mode that works today and an automated deep-learning mode.

Status: v0.1 design · Author: Nicole Lee · November 2026

> **Not a medical device.** This is a research and educational prototype. It does not diagnose, and it does not prescribe treatment. Under UK MHRA rules, software that diagnoses or recommends treatment is a medical device, so a public clinical release would need UKCA marking and clinical validation. That is out of scope.

---

## 1. Requirements

### 1.1 Functional

| ID | Requirement | Mode |
|----|-------------|------|
| F1 | Load a frontal (PA/AP) spinal X-ray image (PNG/JPEG) | both |
| F2 | User marks the upper endplate of the upper end vertebra and the lower endplate of the lower end vertebra (2 points each) | manual |
| F3 | Compute the Cobb angle between the two endplate lines | both |
| F4 | Model predicts 68 landmarks (17 vertebrae T1–L5 × 4 corners) | auto |
| F5 | From landmarks, compute the primary curve plus up to two secondary curves, with the end vertebrae for each | auto |
| F6 | Overlay the lines and landmarks on the image so the user can see why the number was given | both |
| F7 | Let the user drag points to correct them, then recompute | both |
| F8 | Show the severity band and what that range usually means, with signposting to NHS care | both |
| F9 | Flag low-confidence predictions instead of silently returning a number | auto |

### 1.2 Non-functional

| Concern | Target | Why |
|---------|--------|-----|
| Privacy | Manual mode: the image never leaves the browser. Auto mode: processed in memory, never written to disk, no logging of pixels | Medical images are special-category data under UK GDPR |
| Accuracy (auto) | Cobb angle MAE ≤ 5° and SMAPE ≤ 10% on the held-out test split | Manual inter-observer error is about 3–5°, so the model should land within clinician variability |
| Latency (auto) | ≤ 2 s per image on CPU at p95 | One interactive user; no batching needed |
| Cost | £0 to run: static hosting plus one small CPU container | Student project |
| Availability | Best effort; manual mode works offline once loaded | No SLA needed |
| Accessibility | Usable at 400 px phone width; keyboard-operable controls | People will use it on phones |

### 1.3 Constraints

- One developer, part-time, alongside an MSc
- Training on Bristol's GPU cluster or Google Colab; no paid GPUs
- Data: the AASCE 2019 challenge set (609 anterior–posterior X-rays with 68 landmarks each and three Cobb angles), which needs registration
- Stack: Python and PyTorch for ML, FastAPI for serving, plain HTML/JS front end

### 1.4 Explicit non-goals

- Exercise prescription. Curve-specific exercise (e.g. Schroth) depends on curve pattern, rotation and examination, so the tool signposts to physiotherapy instead.
- Lateral views, rotation (Nash–Moe) or Risser grading.
- Tracking a person's curves over time. That needs accounts and storage, which conflicts with the privacy goal.

---

## 2. High-level design

```
                    ┌─────────────────────────────────────────┐
                    │              Browser (web/)             │
                    │                                         │
  X-ray file ──────▶│  Image loader ──▶ Canvas viewer         │
                    │                    │  ▲                 │
                    │       manual taps  │  │ overlay         │
                    │                    ▼  │                 │
                    │               Cobb geometry (JS)        │
                    │                    │                    │
                    │                    ▼                    │
                    │          Severity + guidance panel      │
                    └──────────────┬──────────────────────────┘
                                   │ auto mode only:
                                   │ POST /v1/analyse (image bytes)
                                   ▼
                    ┌─────────────────────────────────────────┐
                    │        Inference API (ml/api.py)        │
                    │  preprocess ──▶ landmark model ──▶      │
                    │  heatmaps ──▶ 68 landmarks ──▶          │
                    │  cobb.py (same maths as the JS) ──▶ JSON│
                    └─────────────────────────────────────────┘
                                   ▲
                                   │ weights.pt (versioned)
                    ┌─────────────────────────────────────────┐
                    │   Offline training (ml/train.py)        │
                    │   AASCE data ──▶ augment ──▶ train ──▶  │
                    │   evaluate.py (MAE, SMAPE per curve)    │
                    └─────────────────────────────────────────┘
```

### 2.1 Components

| Component | Responsibility | Tech |
|-----------|----------------|------|
| `web/index.html` | Viewer, manual marking, result panel, guidance | Vanilla JS and Canvas, no build step |
| `ml/cobb.py` | Landmarks → endplate angles → Cobb angles. Pure numpy, no torch | numpy |
| `ml/dataset.py` | Load AASCE images and landmark CSVs, resize, letterbox, make heatmap targets | numpy, Pillow |
| `ml/model.py` | U-Net-style network outputting 68 heatmaps | PyTorch |
| `ml/train.py` | Training loop, checkpointing | PyTorch |
| `ml/evaluate.py` | Landmark error, Cobb MAE, SMAPE on the test split | numpy |
| `ml/api.py` | HTTP inference service | FastAPI, uvicorn |

### 2.2 Data flow (auto mode)

1. The browser sends the image as `multipart/form-data` to `POST /v1/analyse`.
2. The API converts it to greyscale, letterboxes it to 1024 × 512 (H × W) and normalises it.
3. The model outputs 68 heatmaps at ¼ resolution. Each landmark is the soft-argmax of its heatmap, and the peak value is kept as its confidence.
4. Landmarks are mapped back to original image coordinates.
5. `cobb.py` computes the curves.
6. The JSON response goes back; the browser draws the overlay and lets the user drag landmarks. A drag recomputes in JS, so no second round-trip.

---

## 3. Deep dive

### 3.1 Cobb angle maths (shared by JS and Python)

For an endplate through points **a** and **b**, its tilt is

  θ = atan2(b_y − a_y, b_x − a_x), normalised to (−90°, 90°].

The Cobb angle between endplates *i* and *j* is |θ_i − θ_j|.

**Manual mode:** one upper and one lower endplate are marked, so Cobb = |θ_upper − θ_lower|.

**Auto mode:** 17 vertebrae give 34 endplates in top-to-bottom order.
- Primary curve: the pair (i, j) with i < j that maximises |θ_i − θ_j|. Those endplates identify the end vertebrae.
- Secondary curves: the same search restricted to endplates above i, and to endplates below j.

This mirrors the method used by the AASCE organisers, simplified to endplates only.

### 3.2 Model

- **Approach: heatmap regression**, not direct regression of angles or coordinates. Heatmaps keep spatial structure, are the standard for anatomical landmarks, and give a natural confidence signal (peak height).
- **Architecture:** a U-Net with 4 down-sampling stages, 68 output channels at ¼ input resolution. A pretrained ResNet-34 encoder is a planned upgrade (see §6).
- **Target:** a Gaussian with σ = 2 px at output resolution for each landmark.
- **Loss:** MSE on heatmaps, plus an optional auxiliary loss on the Cobb angle computed from soft-argmax coordinates, which directly penalises the error clinicians care about.
- **Augmentation:** small rotations (±10°), scale (0.9–1.1), brightness and contrast jitter, CLAHE. **No horizontal flip** unless landmark left/right labels are swapped too, otherwise curve direction is corrupted.

### 3.3 API contract

`POST /v1/analyse`

Request: `multipart/form-data` with field `image` (PNG/JPEG, ≤ 10 MB).

Response `200`:

```json
{
  "model_version": "unet-v0.1",
  "image_size": {"width": 1200, "height": 2400},
  "landmarks": [[x, y], "... 68 points in image pixels, order TL, TR, BL, BR per vertebra, T1 to L5"],
  "landmark_confidence": [0.91, "... 68 values in [0, 1]"],
  "curves": [
    {"name": "primary", "cobb_deg": 42.3,
     "upper_endplate": 9, "lower_endplate": 25,
     "upper_vertebra": "T5", "lower_vertebra": "T12"}
  ],
  "low_confidence": false,
  "disclaimer": "Research prototype. Not a diagnosis."
}
```

Errors: `400` not an image or unreadable, `413` too large, `422` image too small or wrong aspect for a full-spine view, `503` model not loaded. Error bodies are `{"error": "<code>", "message": "<what to do>"}`.

`GET /healthz` returns `{"status": "ok", "model_loaded": true}`.

### 3.4 Storage

Nothing user-related is stored on a server. Manual mode saves the user's own X-ray (the original file), crop, curves and display settings in the browser's IndexedDB on their device, so the next visit picks up where they left off; "Forget saved X-ray" deletes it. It never syncs anywhere, so it's per browser and per device and can be cleared by the browser. Model weights live in the repo's release assets (or Git LFS) with a version tag that the API reports as `model_version`.

### 3.5 Error handling

- Low confidence: if mean landmark confidence < 0.3 or any vertebra's four corners collapse within 3 px, set `low_confidence: true`. The front end then shows "Check the points" and asks the user to correct them.
- Out-of-distribution images (a chest X-ray, a photo of a screen): the aspect ratio check catches some; confidence catches most of the rest.
- No retries are needed; requests are idempotent and the user can resubmit.

---

## 4. Scale and reliability

- **Load:** a portfolio project expects < 100 requests a day. One CPU container (e.g. a free tier on Hugging Face Spaces or Render) is enough. The model is about 30 MB and inference takes about 1 s on CPU.
- **Scaling:** stateless API, so it scales horizontally if ever needed. Manual mode puts no load on the server.
- **Monitoring:** log request count, latency, `low_confidence` rate and error codes. Never log pixels. A rising low-confidence rate is the drift alarm.

---

## 5. Trade-offs

| Decision | Chosen | Alternative | Why |
|----------|--------|-------------|-----|
| Ship manual mode first | Yes | Wait for the model | Useful today, needs no data, and gives a baseline to compare the model against |
| Heatmaps | Yes | Regress the 3 angles directly | Explainable overlay, user can correct points, confidence comes free. Direct regression is simpler but a black box |
| Client-side geometry | Duplicated in JS and Python | Server only | Instant recompute on drag; manual mode works offline. Cost: two implementations to keep in sync, mitigated by shared test cases |
| Vanilla JS | Yes | React | One page, no build step, nothing to maintain. Revisit if the UI grows |
| On-device storage only | IndexedDB in the browser | Accounts and server storage | Privacy and no GDPR overhead; no re-uploading. Cost: no sync across devices, and a progression history would also stay on one device |
| FastAPI on CPU | Yes | Serverless GPU | Free and fast enough for one user at a time |

---

## 6. What to revisit as it grows

1. **Pretrained encoder** (ResNet-34 or ConvNeXt via `timm`) once the baseline U-Net results are in.
2. **Browser inference** via ONNX Runtime Web, so auto mode also keeps the image on-device. This removes the server entirely and is the strongest privacy story.
3. **Second dataset** for external validation. AASCE images come from one source; a model can look good on it and fail elsewhere.
4. **Uncertainty**: Monte Carlo dropout or a small ensemble to give an angle ± range, not just a confidence flag.
5. **Curve type labels** (thoracic, thoracolumbar, lumbar) from the end-vertebra positions.
6. **Clinical framing**: if this ever went beyond a prototype, talk to a clinician and look at MHRA software-as-a-medical-device guidance before anything else.

---

## 7. Evaluation plan

| Metric | Definition | Target |
|--------|------------|--------|
| Landmark MRE | Mean Euclidean error over 68 points, in pixels at original resolution | Reported, no target |
| Cobb MAE | Mean absolute error per curve, in degrees | ≤ 5° |
| SMAPE | AASCE challenge metric over the 3 angles | ≤ 10% |
| Agreement | Bland–Altman plot, model vs ground truth | Bias near 0, limits within ±10° |
| Band accuracy | Severity band (mild/moderate/severe) correct | Reported |

Baseline to beat: the manual mode, measured by having the same person mark 20 test images twice, which gives an intra-observer error.
