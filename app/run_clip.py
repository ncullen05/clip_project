# app/run_clip.py
"""
Main entry point for the CLIP-based aesthetics scoring system.

Loads a pre-trained CLIP model, initializes the scorer with 
aesthetic evaluation prompts, and scores a sample image.
"""

import json
import sys

from app.model import CLIPModel
from app.scorer import ClipAestheticsScorer
from app.prompt_registry import get_prompt_sets

def main():
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python -m app.run_clip <path_to_image>")

    image_path = sys.argv[1]

    
    # Load aesthetic evaluation prompts from the prompt registry
    pos, neg = get_prompt_sets()

    # Initialize CLIP model (automatically selects GPU if available)
    clip_model = CLIPModel()
    # Create scorer with prompt sets and requested number of top results
    scorer = ClipAestheticsScorer(clip_model, pos, neg, top_k=3)

    # Score a sample image - accepts file path, bytes, or PIL Image
    results = scorer.score(image_path)
    # Display scoring results
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()