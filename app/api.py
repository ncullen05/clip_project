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


def parse_features(features_str: str) -> list[str]:
    """
    Expects a comma-separated list of feature keys, e.g.:
    "lighting/exposure,contrast"
    """
    features = [f.strip() for f in features_str.split(",") if f.strip()]
    return features


@app.get("/health") # When someone sends an HTTP GET request to /health, run this function.
def health() -> Dict[str, str]:
    return {"status": "ok", "model": getattr(clip_model, "model_name", "unknown")}


@app.post("/score") # When someone sends an HTTP POST request to /score, run this function.
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

    requested = parse_features(features)

    # Run scorer (your CLIPModel supports bytes input)
    result: Dict[str, Any] = scorer.score(img_bytes)

    # Optional filtering to selected features
    if requested is not None:
        all_features = result.get("features", {})
        filtered = {k: all_features[k] for k in requested if k in all_features}
        result["features"] = filtered

    return JSONResponse(content=result)