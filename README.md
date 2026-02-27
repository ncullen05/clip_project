# Urban Aesthetics — CLIP Scoring Backend (Python)

This repository contains the Python CLIP-based scoring backend for the Urban Aesthetics project.

It uses OpenAI CLIP to extract feature-level aesthetic signals from an image (lighting/exposure, contrast, framing/perspective, visual focus, visual complexity) and returns a stable JSON schema.

## Design principles

- This repository produces objective, feature-level measurements derived from a CLIP model.
- The output consists only of numeric scores and prompt-alignment evidence.
- No semantic interpretation, advice, or qualitative judgment is generated in this codebase.
- All outputs are deterministic given the model, prompts, and calibration constants.

Any interpretation, explanation, or user-facing feedback must be performed by a separate system using this output as input.

---

## Output schema

The scorer returns JSON in the format documented in `SCHEMA.md`.

### Top-level structure

```json
{
  "schema_version": "1.0",
  "model": "ViT-B/16",
  "features": { ... }
}
```

### Per-feature structure

```json
{
  "score_0_10": 6.42,
  "avg_positive": 0.245,
  "avg_negative": 0.238,
  "delta": 0.007,
  "top_positive": [
    { "prompt": "...", "score": 0.27 }
  ],
  "top_negative": [
    { "prompt": "...", "score": 0.25 }
  ]
}
```

- `score_0_10` is a calibrated value derived from `delta`.
- All other fields represent raw CLIP similarity measurements.

### Feature keys (public API)

The following feature keys are fixed and must remain consistent across prompts, calibration, and output:

- `lighting/exposure`
- `contrast`
- `framing/perspective`
- `visual focus`
- `visual complexity`

## Repository structure

- `app/model.py` — Loads the CLIP model, selects device (CPU/GPU), converts inputs to RGB PIL images, and encodes image embeddings.
- `app/scorer.py` — Caches prompt tokens and text embeddings; computes per-feature deltas, calibrated scores, and top-K prompt evidence.
- `app/mapping.py` — Delta to 0–10 mapping logic and per-feature calibration constants.
- `app/prompt_registry.py` — Loads positive and negative prompt sets for each feature.
- `app/run_clip.py` — CLI entry point for scoring a single image and printing JSON output.
- `app/calibrate.py` — Calibration script for computing per-feature percentile ranges from a calibration image set.

## Setup (Windows PowerShell)

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run scoring on a single image:

```powershell
python -m app.run_clip <path_to_image>
```

Example:

```powershell
python -m app.run_clip images\example.jpg
```

The command prints JSON only to stdout.

## Calibration workflow

CLIP similarity deltas are relative and unbounded. Calibration maps these deltas onto a meaningful 0–10 scale.

1. Place calibration images in a local folder (commonly `images/`).
2. Do not commit calibration images; ensure the folder is ignored by `.gitignore`.
3. Run the calibration script:

```powershell
python -m app.calibrate
```

The script prints per-feature calibration values:

- `low_delta` (5th percentile)
- `high_delta` (95th percentile)

Copy these values into `app/mapping.py`.

## Important notes

- Calibration is model-specific (for example, `ViT-B/16` vs `ViT-B/32`).
- Calibration is prompt-set specific.
- Any change to the CLIP model variant, prompt wording, or number of prompts requires recalibration.