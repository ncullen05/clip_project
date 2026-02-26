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