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
    """Provider-agnostic adapter for generating text-only interpretation."""

    @abstractmethod
    def feature_feedback(self, feature_key: str, evidence: FeatureEvidence) -> FeatureFeedback:
        raise NotImplementedError


class RuleBasedSuggestionsProvider(SuggestionsProvider):
    """Fallback provider that generates deterministic evidence-grounded text."""

    def feature_feedback(self, feature_key: str, evidence: FeatureEvidence) -> FeatureFeedback:
        positives = [p.prompt for p in evidence.top_positive[:2]]
        negatives = [p.prompt for p in evidence.top_negative[:2]]

        if positives:
            positive_part = f"Positive evidence emphasizes {', '.join(positives)}"
        else:
            positive_part = "Positive evidence is limited"

        if negatives:
            negative_part = f"while negative evidence highlights {', '.join(negatives)}"
        else:
            negative_part = "and there are no strong negative prompt matches"

        summary = f"For {feature_key}, {positive_part}, {negative_part}."

        suggestions = self._build_suggestions(feature_key, positives, negatives)
        print(f"[suggest][fallback] {feature_key} summary={summary}")
        print(f"[suggest][fallback] {feature_key} suggestions={suggestions}")
        return FeatureFeedback(summary=summary, suggestions=suggestions)

    def _build_suggestions(
        self,
        feature_key: str,
        positives: List[str],
        negatives: List[str],
    ) -> List[str]:
        suggestions: List[str] = []

        if positives:
            suggestions.append(
                f"Keep the elements that align with {feature_key} strengths, especially cues similar to: {positives[0]}."
            )
        else:
            suggestions.append(
                f"Introduce clearer visual cues for {feature_key} so positive descriptors appear more consistently."
            )

        if negatives:
            suggestions.append(
                f"Reduce traits associated with: {negatives[0]}, and replace them with cleaner alternatives."
            )
        else:
            suggestions.append(
                "Preserve the current balance and avoid adding conflicting visual elements."
            )

        suggestions.append(
            f"Iterate on {feature_key} using small composition or styling adjustments, then re-measure to compare evidence shifts."
        )

        return suggestions[:3]


class OpenAISuggestionsProvider(SuggestionsProvider):
    """OpenAI-backed provider for text-only, evidence-based feature feedback."""

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
            content = response.output_text
            parsed = json.loads(content)
            summary = str(parsed.get("summary", "")).strip()
            suggestions = parsed.get("suggestions", [])
            if not isinstance(suggestions, list):
                suggestions = []
            cleaned = [str(s).strip() for s in suggestions if str(s).strip()][:3]
            while len(cleaned) < 3:
                cleaned.append("Refine the visual choices and re-run measurement to compare evidence changes.")

            print(f"[suggest][openai] {feature_key} summary={summary}")
            print(f"[suggest][openai] {feature_key} suggestions={cleaned}")
            return FeatureFeedback(summary=summary, suggestions=cleaned)
        except Exception as exc:
            print(f"[suggest][openai][error] feature={feature_key} error={exc}")
            fallback = RuleBasedSuggestionsProvider().feature_feedback(feature_key, evidence)
            return fallback