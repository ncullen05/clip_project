# app/api.py
from __future__ import annotations 

import json
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from app.model import CLIPModel
from app.scorer import ClipAestheticsScorer
from app.prompt_registry import get_prompt_sets

# Create instance of FastAPI app
app = FastAPI(title="Urban Aesthetics CLIP API", version="1.0")

# Load once at startup 
pos_prompts, neg_prompts = get_prompt_sets()
clip_model = CLIPModel()  
scorer = ClipAestheticsScorer(clip_model, pos_prompts, neg_prompts, top_k=3)


def parse_features(features: str) -> Optional[List[str]]:
    if features is None:
        return None

    s = features.strip()

    # Strip a single pair of wrapping quotes if present
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1].strip()

    if not s:
        return None

    out = []
    for part in s.split(","):
        key = part.strip().strip('"').strip("'")  # also strip any stray quotes per-item
        if key:
            out.append(key)

    return out or None


@app.get("/health") # When someone sends an HTTP GET request to /health, run this function.
def health() -> Dict[str, str]:
    return {"status": "ok", "model": getattr(clip_model, "model_name", "unknown")}


@app.post("/score")
async def score_image(
    image: UploadFile = File(...),
    features: str = Form(...),
) -> JSONResponse:
    # Basic content-type check
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    # --- DEBUG: show exactly what Android sent ---
    print("RAW features string:", repr(features))

    requested = parse_features(features)

    # --- DEBUG: show parsed list ---
    print("Parsed requested list:", requested)

    # Run scorer
    result: Dict[str, Any] = scorer.score(img_bytes)

    # --- DEBUG: show what backend has available before filtering ---
    all_features = result.get("features", {})
    print("Backend feature keys available:", list(all_features.keys()))

    # Optional filtering to selected features
    if requested is not None:
        missing = [k for k in requested if k not in all_features]
        if missing:
            print("Requested keys missing from backend result:", missing)

        filtered = {k: all_features[k] for k in requested if k in all_features}
        result["features"] = filtered

    return JSONResponse(content=result)