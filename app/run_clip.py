# app/run_clip.py
"""
Main entry point for the CLIP-based aesthetics scoring system.

Loads a pre-trained CLIP model, initializes the scorer with 
aesthetic evaluation prompts, and scores a sample image.
"""

from app.model import CLIPModel
from app.scorer import ClipAestheticsScorer
from app.prompt_registry import get_prompt_sets

def main():
    """
    Main function demonstrating image aesthetic scoring workflow.
    
    Steps:
    1. Load positive and negative aesthetic evaluation prompts
    2. Initialize CLIP model for encoding images and text
    3. Create scorer with prompt sets (builds token and feature caches)
    4. Score a sample image across all aesthetic dimensions
    5. Print results showing aesthetic evaluation scores
    """
    # Load aesthetic evaluation prompts from the prompt registry
    pos, neg = get_prompt_sets()

    # Initialize CLIP model (automatically selects GPU if available)
    clip_model = CLIPModel()
    # Create scorer with prompt sets and requested number of top results
    scorer = ClipAestheticsScorer(clip_model, pos, neg, top_k=3)

    # Score a sample image - accepts file path, bytes, or PIL Image
    results = scorer.score("images/alfie.jpg")
    # Display scoring results
    print(results)

if __name__ == "__main__":
    main()