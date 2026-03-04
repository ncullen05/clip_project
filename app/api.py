# app/api.py
from __future__ import annotations

import os
from typing import Optional, List, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from app.model import CLIPModel
from app.scorer import ClipAestheticsScorer
from app.prompt_registry import get_prompt_sets
from app.suggestions import (
    ALLOWED_FEATURE_KEYS,
    OpenAISuggestionsProvider,
    SuggestRequest,
    SuggestResponse,
    SuggestionProviderError,
    SuggestionsProvider,
)

# Create instance of FastAPI app
app = FastAPI(title="Urban Aesthetics CLIP API", version="1.0")

def _suggest_log(message: str) -> None:
    print(message, flush=True)

# Load once at startup
pos_prompts, neg_prompts = get_prompt_sets()
clip_model = CLIPModel()
scorer = ClipAestheticsScorer(clip_model, pos_prompts, neg_prompts, top_k=3)

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if openai_api_key:
    suggestions_provider: SuggestionsProvider | None = OpenAISuggestionsProvider(model=openai_model)
    _suggest_log(f"[suggest] Using OpenAI provider model={openai_model}")
else:
    suggestions_provider = None
    _suggest_log("[suggest] OPENAI_API_KEY not set; /suggest will return 503")


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


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model": getattr(clip_model, "model_name", "unknown")}


@app.post("/score")
async def score_image(
    image: UploadFile = File(...),
    features: str = Form(...),
) -> JSONResponse:
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    requested = parse_features(features)
    result: Dict[str, object] = scorer.score(img_bytes)

    if requested is not None:
        all_features = result.get("features", {})
        filtered = {k: all_features[k] for k in requested if k in all_features}
        result["features"] = filtered

    return JSONResponse(content=result)


@app.post("/suggest", response_model=SuggestResponse)
def suggest(payload: SuggestRequest) -> SuggestResponse:
    if suggestions_provider is None:
        _suggest_log("[suggest][error] OPENAI_API_KEY not set")
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not set")

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

    feature_feedback = {}
    if not selected_features:
        _suggest_log("[suggest] no valid features selected")
    for key, evidence in selected_features.items():
        try:
            feedback = suggestions_provider.feature_feedback(key, evidence)
        except SuggestionProviderError as exc:
            _suggest_log(f"[suggest][error] {exc.detail}")
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

        _suggest_log(f"[suggest] feature={key}")
        _suggest_log(
            "raw_scores: "
            f"score_0_10={evidence.score_0_10}, "
            f"avg_positive={evidence.avg_positive}, "
            f"avg_negative={evidence.avg_negative}, "
            f"delta={evidence.delta}"
        )
        for i, prompt_score in enumerate(evidence.top_positive, start=1):
            _suggest_log(f"raw_top_positive {i}: {prompt_score.prompt} => {prompt_score.score}")
        for i, prompt_score in enumerate(evidence.top_negative, start=1):
            _suggest_log(f"raw_top_negative {i}: {prompt_score.prompt} => {prompt_score.score}")

        _suggest_log(f"summary: {feedback.summary}")
        for i, suggestion in enumerate(feedback.suggestions, start=1):
            _suggest_log(f"suggestion {i}: {suggestion}")
        feature_feedback[key] = feedback

    return SuggestResponse(
        schema_version="1.0",
        model="ViT-B/16",
        feature_feedback=feature_feedback,
    )