# app/api.py
from __future__ import annotations

import os
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from app.model import CLIPModel
from app.scorer import ClipAestheticsScorer
from app.prompt_registry import get_prompt_sets
from app.suggestions import FeatureEvidence, PromptScore, FeatureFeedback
from app.suggestions import (
    ALLOWED_FEATURE_KEYS,
    OpenAISuggestionsProvider,
    RuleBasedSuggestionsProvider,
    SuggestRequest,
    SuggestResponse,
    SuggestionsProvider,
)

# Create instance of FastAPI app
app = FastAPI(title="Urban Aesthetics CLIP API", version="1.0")

# Load once at startup 
pos_prompts, neg_prompts = get_prompt_sets()
clip_model = CLIPModel()  
scorer = ClipAestheticsScorer(clip_model, pos_prompts, neg_prompts, top_k=3)

if os.getenv("OPENAI_API_KEY"):
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    suggestions_provider: SuggestionsProvider = OpenAISuggestionsProvider(model=openai_model)
    print(f"[suggest] Using OpenAI provider model={openai_model}")
else:
    suggestions_provider = RuleBasedSuggestionsProvider()
    print("[suggest] OPENAI_API_KEY missing; using fallback rule-based provider")


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

def _evidence_from_result(feature_result: Dict[str, Any]) -> FeatureEvidence:
    return FeatureEvidence(
        score_0_10=float(feature_result.get("score_0_10")),
        avg_positive=feature_result.get("avg_positive"),
        avg_negative=feature_result.get("avg_negative"),
        delta=feature_result.get("delta"),
        top_positive=[
            PromptScore(prompt=p.get("prompt", ""), score=float(p.get("score", 0.0)))
            for p in (feature_result.get("top_positive") or [])
        ],
        top_negative=[
            PromptScore(prompt=p.get("prompt", ""), score=float(p.get("score", 0.0)))
            for p in (feature_result.get("top_negative") or [])
        ],
    )

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
        all_features = result.get("features", {})

        filtered = {k: all_features[k] for k in requested if k in all_features}
        result["features"] = filtered

 # --- DEBUG: generate suggestions from evidence and print to terminal ---
    try:
        for key, fr in result.get("features", {}).items():
            if key not in ALLOWED_FEATURE_KEYS:
                continue
            evidence = _evidence_from_result(fr)
            feedback = suggestions_provider.feature_feedback(key, evidence)

            print(f"\n[suggest][from_score] feature={key}")
            print(f"[suggest][from_score] summary: {feedback.summary}")
            for i, s in enumerate(feedback.suggestions, start=1):
                print(f"[suggest][from_score] suggestion {i}: {s}")
    except Exception as exc:
        print(f"[suggest][from_score][error] {exc}")

    return JSONResponse(content=result)

@app.post("/suggest", response_model=SuggestResponse)
def suggest(payload: SuggestRequest) -> SuggestResponse:
    features = payload.features
    if payload.requested_features is not None:
        requested = [key for key in payload.requested_features if key in ALLOWED_FEATURE_KEYS]
    else:
        requested = [key for key in features.keys() if key in ALLOWED_FEATURE_KEYS]

    selected_features = {
        key: features[key]
        for key in requested
        if key in features and key in ALLOWED_FEATURE_KEYS
    }

    print(f"[suggest] processing_features={list(selected_features.keys())}")

    feature_feedback = {}
    for key, evidence in selected_features.items():
        try:
            feature_feedback[key] = suggestions_provider.feature_feedback(key, evidence)
        except Exception as exc:
            print(f"[suggest][error] feature={key} error={exc}")
            feature_feedback[key] = RuleBasedSuggestionsProvider().feature_feedback(key, evidence)

    return SuggestResponse(
        schema_version="1.0",
        model="ViT-B/16",
        feature_feedback=feature_feedback,
    )