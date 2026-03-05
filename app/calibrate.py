# app/calibrate.py
"""
Calibration script for CLIP aesthetic scoring.

Runs the scorer over a folder of calibration images and computes
per-feature delta distributions. Uses percentiles to recommend
(low_delta, high_delta) for mapping delta -> 0..10.
"""

import os
import json
import numpy as np

from app.model import CLIPModel
from app.scorer import ClipAestheticsScorer
from app.prompt_registry import get_prompt_sets

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(folder: str) -> list[str]:
    paths = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name.lower())[1]
        if ext in IMAGE_EXTS:
            paths.append(os.path.join(folder, name))
    return sorted(paths)


def main():
    # Folder containing ALL calibration images
    # (Unsplash + COCO + phone photos together)
    folder = "images"

    paths = list_images(folder)
    if not paths:
        raise RuntimeError(f"No images found in '{folder}'")

    # Load prompts and scorer
    pos, neg = get_prompt_sets()
    clip_model = CLIPModel()
    scorer = ClipAestheticsScorer(clip_model, pos, neg)

    # Collect deltas per feature
    deltas_by_feature: dict[str, list[float]] = {k: [] for k in pos.keys()}

    for i, path in enumerate(paths, start=1):
        results = scorer.score(path)
        for feature_key in deltas_by_feature:
            deltas_by_feature[feature_key].append(results["features"][feature_key]["delta"])
        print(f"{i}/{len(paths)}", end="\r")

    print()  # newline after progress

    # Compute calibration ranges using percentiles
    calibration = {}
    for feature_key, deltas in deltas_by_feature.items():
        arr = np.asarray(deltas, dtype=np.float64)

        low = float(np.percentile(arr, 5))    # maps to score 0
        high = float(np.percentile(arr, 95))  # maps to score 10

        calibration[feature_key] = {
            "low_delta": low,
            "high_delta": high,
            "count": int(arr.size),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }

    # Print as JSON for easy copy-paste into schema.py
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()