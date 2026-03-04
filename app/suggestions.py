from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, validator


ALLOWED_FEATURE_KEYS = {
    "lighting/exposure",
    "contrast",
    "framing/perspective",
    "visual focus",
    "visual complexity",
}


class PromptScore(BaseModel):
    prompt: str
    score: float


class FeatureEvidence(BaseModel):
    score_0_10: float
    avg_positive: Optional[float] = None
    avg_negative: Optional[float] = None
    delta: Optional[float] = None
    top_positive: List[PromptScore] = Field(default_factory=list)
    top_negative: List[PromptScore] = Field(default_factory=list)


class SuggestRequest(BaseModel):
    schema_version: str
    model: str
    features: Dict[str, FeatureEvidence]
    requested_features: Optional[List[str]] = None

    @validator("schema_version")
    def validate_schema_version(cls, value: str) -> str:
        if value != "1.0":
            raise ValueError("schema_version must be '1.0'")
        return value

    @validator("model")
    def validate_model(cls, value: str) -> str:
        if value != "ViT-B/16":
            raise ValueError("model must be 'ViT-B/16'")
        return value


class FeatureFeedback(BaseModel):
    summary: str
    suggestions: List[str]


class SuggestResponse(BaseModel):
    schema_version: str = "1.0"
    model: str = "ViT-B/16"
    feature_feedback: Dict[str, FeatureFeedback]


class SuggestionsProvider(ABC):
    @abstractmethod
    def feature_feedback(self, feature_key: str, evidence: FeatureEvidence) -> FeatureFeedback:
        raise NotImplementedError


class SuggestionProviderError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class OpenAISuggestionsProvider(SuggestionsProvider):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def feature_feedback(self, feature_key: str, evidence: FeatureEvidence) -> FeatureFeedback:
        try:
            payload = {
                "feature_key": feature_key,
                "evidence": evidence.dict(),
            }
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "You generate feedback from CLIP evidence only. "
                                    "Never claim to have seen an image. "
                                    "Do not include numeric scores in suggestions. "
                                    "Output strict JSON with keys: summary (string), suggestions (array of exactly 3 strings)."
                                ),
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": json.dumps(payload)}],
                    },
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "feature_feedback",
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "summary": {"type": "string"},
                                "suggestions": {
                                    "type": "array",
                                    "minItems": 3,
                                    "maxItems": 3,
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["summary", "suggestions"],
                        },
                    }
                },
            )
            parsed = json.loads(response.output_text)
            summary = str(parsed.get("summary", "")).strip()
            suggestions = parsed.get("suggestions", [])
            if not summary or not isinstance(suggestions, list) or len(suggestions) != 3:
                raise ValueError("Schema mismatch")
            cleaned = [str(s).strip() for s in suggestions]
            if any(not s for s in cleaned):
                raise ValueError("Schema mismatch")

            return FeatureFeedback(summary=summary, suggestions=cleaned)
        except json.JSONDecodeError as exc:
            raise SuggestionProviderError(
                status_code=502,
                detail=f"feature={feature_key}; LLM returned invalid JSON",
            ) from exc
        except ValueError as exc:
            raise SuggestionProviderError(
                status_code=502,
                detail=f"feature={feature_key}; LLM returned invalid JSON",
            ) from exc
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if isinstance(status_code, int) and 400 <= status_code <= 599:
                raise SuggestionProviderError(
                    status_code=502,
                    detail=f"feature={feature_key}; OpenAI {status_code}",
                ) from exc
            raise SuggestionProviderError(
                status_code=502,
                detail=f"feature={feature_key}; OpenAI request failed",
            ) from exc