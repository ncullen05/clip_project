# Urban Aesthetics CLIP Output Schema

Schema version: 1.0

This document defines the JSON output produced by the CLIP-based
Urban Aesthetics scoring system.

This schema is treated as a **stable API contract** between:
- the Python scoring backend
- the Android application
- the LLM explanation service

---

## Top-level object

```json
{
  "schema_version": "1.0",
  "model": "ViT-B/16",
  "features": { ... }
}
```

## Suggest endpoint contract (`POST /suggest`)

Request body (text-only evidence):

```json
{
  "schema_version": "1.0",
  "model": "ViT-B/16",
  "features": {
    "lighting/exposure": {
      "score_0_10": 6.4,
      "avg_positive": 0.24,
      "avg_negative": 0.23,
      "delta": 0.01,
      "top_positive": [{ "prompt": "...", "score": 0.25 }],
      "top_negative": [{ "prompt": "...", "score": 0.24 }]
    }
  },
  "requested_features": ["lighting/exposure"]
}
```

Response body:

```json
{
  "schema_version": "1.0",
  "model": "ViT-B/16",
  "feature_feedback": {
    "lighting/exposure": {
      "summary": "<string>",
      "suggestions": ["<string>", "<string>", "<string>"]
    }
  }
}
```

Behavior notes:
- The endpoint consumes evidence only and does not receive image bytes.
- Unknown feature keys are ignored.
- If `requested_features` is provided, output is limited to those known keys.