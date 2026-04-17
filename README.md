# Urban Aesthetics — CLIP Scoring Backend (Python)

This repository contains the Python CLIP-based scoring backend for the Urban Aesthetics project.

This backend uses OpenAI CLIP (ViT-B/16) to measure feature-level photographic quality signals from an image and returns a stable JSON contract used by the mobile UI. It also includes an optional /suggest endpoint that sends only structured evidence (never the image) to an LLM to generate user-friendly feedback.

## Core rule:

CLIP = measurement (deterministic)  
LLM = interpretation (text-only, evidence-only; no image is ever sent)

## Features measured (locked keys)

These feature keys are part of the stable API contract and are used consistently across backend + Android:
- `lighting/exposure`
- `contrast`
- `framing/perspective`
- `visual focus`
- `visual complexity`

## How scoring works

For each feature:
- The image is encoded once with CLIP.
- Prompts for that feature are encoded once and cached.
- Cosine similarity is computed between the image embedding and each prompt embedding.

We compute:
- avg_positive (mean similarity over positive prompts)
- avg_negative (mean similarity over negative prompts)
- delta = avg_positive - avg_negative
- delta is mapped to a calibrated score_0_10 using bounds learned from a calibration dataset.

## Calibration bounds:

5th percentile → 0  
95th percentile → 10

If the model or prompt set is changed, recalibration is necessary.

## Repository Layout

```
clip_project/
├─ app/
│  ├─ api.py                # FastAPI app + endpoints
│  ├─ model.py              # CLIP model loading / image encoding helpers
│  ├─ scorer.py             # CLIP scoring logic + top_k prompt evidence
│  ├─ prompt_registry.py    # Loads prompt sets for all features
│  ├─ prompts/              # Prompt sets (per feature)
│  ├─ mapping.py            # Calibration ranges + delta→score mapping
│  ├─ calibrate.py          # Calibration script (writes/updates ranges)
│  └─ suggestions.py        # /suggest request/response models + OpenAI provider
├─ requirements.txt
├─ README.md
├─ SCHEMA.md                # Canonical response schema / contract notes
└─ THIRD_PARTY_NOTICES.md
```

## Requirements

- Python 3.10+ recommended (3.11+ ideal)
- A working PyTorch install supported by your machine (CPU or CUDA)
- Git (required because requirements.txt installs CLIP from GitHub)

## Install for local Development (Windows PowerShell)

Create and activate a virtual environment:
- `python -m venv .venv`
- `.venv\Scripts\Activate.ps1`

Install dependencies:
- `pip install -r requirements.txt`

Note: this installs OpenAI CLIP from GitHub as pinned in requirements.txt

## Run the API

Start FastAPI with Uvicorn
- `uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload`

`--reload` is for development only.  
Use `0.0.0.0` so a mobile device can reach the server on your LAN IP.

## API Endpoints

### Health check

**GET /health**

Response:

```json
{
  "status": "ok",
  "model": "ViT-B/16"
}
```

### Score an image

**POST /score**  
**Content-Type:** multipart/form-data

Form fields:

- image: JPG/PNG file
- features: comma-separated string of feature keys

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/score" \
  -F "image=@example.jpg" \
  -F "features=lighting/exposure,contrast,visual focus"
```

Response (schema v1.0 — stable):

```json
{
  "schema_version": "1.0",
  "model": "ViT-B/16",
  "features": {
    "lighting/exposure": {
      "score_0_10": 5.8,
      "avg_positive": 0.19,
      "avg_negative": 0.20,
      "delta": -0.01,
      "top_positive": [{"prompt":"...", "score":0.21}],
      "top_negative": [{"prompt":"...", "score":0.22}]
    }
  }
}
```

### Generate suggestions (text-only, evidence-only)

**POST /suggest**  
**Content-Type:** application/json

Request body:

The same feature evidence object returned from /score

A list of requested features

Example structure:

```json
{
  "schema_version": "1.0",
  "model": "ViT-B/16",
  "features": { "...same as /score..." },
  "requested_features": ["lighting/exposure", "contrast"]
}
```

Response:

```json
{
  "schema_version": "1.0",
  "model": "gpt-4o-mini",
  "feature_feedback": {
    "lighting/exposure": {
      "summary": "…",
      "suggestions": ["…", "…", "…"]
    }
  }
}
```

**Important privacy guarantee:**  
The /suggest endpoint never receives the image—only structured CLIP evidence.