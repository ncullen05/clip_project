# app/scorer.py
"""
CLIP-based Aesthetics Scorer for evaluating image quality across multiple dimensions.

This module provides a scoring system that uses CLIP's vision-language alignment to
evaluate images across five aesthetic dimensions: lighting/exposure, contrast,
framing/perspective, visual focus, and visual complexity. Each dimension uses
positive and negative prompt sets to measure how well an image aligns with good
aesthetic practices.
"""

import torch
import clip
import numpy as np

class ClipAestheticsScorer:
    """
    Scores images across multiple aesthetic dimensions using CLIP embeddings.
    
    Uses pre-computed text embeddings (caches) for positive and negative prompts
    to evaluate images efficiently. For each dimension, computes similarity scores
    between the image and both positive/negative prompts, then ranks the top matches.
    """
    def __init__(self, clip_model, positive_prompts: dict, negative_prompts: dict, top_k: int = 3):
        """
        Initialize the aesthetics scorer with CLIP model and prompt sets.
        
        Builds persistent caches for tokenized prompts and their encoded features
        during initialization. This one-time setup cost is amortized over many
        scoring calls, making subsequent image evaluations very fast.
        
        Args:
            clip_model: CLIPModel instance for encoding images and text.
            positive_prompts: Dict mapping aesthetic dimensions to lists of positive
                            prompts (e.g., high quality attribute descriptions).
            negative_prompts: Dict mapping aesthetic dimensions to lists of negative
                            prompts (e.g., low quality attribute descriptions).
            top_k: Number of top-scoring prompts to return per dimension (default: 3).
        """
        # Store references to the CLIP model and its components
        self.clip_model = clip_model
        self.model = clip_model.model
        self.device = clip_model.device

        # Store the prompt dictionaries
        self.positive_prompts = positive_prompts
        self.negative_prompts = negative_prompts
        # Number of top-scoring prompts to return in results
        self.top_k = top_k

        # Build and store persistent caches to avoid recomputing on every image score
        # Token cache: pre-tokenized prompts ready for encoding
        self._token_cache = self._build_token_cache()
        # Feature cache: pre-encoded and normalized text embeddings
        self._text_feat_cache = self._build_text_feature_cache()

    def _build_token_cache(self) -> dict:
        """
        Pre-tokenize all prompts and cache them for reuse.
        
        Tokenization converts text prompts to token sequences that CLIP understands.
        Since tokenization can be computationally expensive and we use the same
        prompts for every image, we tokenize once during initialization and reuse
        the tokens for all subsequent scoring calls.
        
        Returns:
            dict: Nested dictionary mapping aesthetic dimension -> {"pos": tokens, "neg": tokens}.
                 Tokens are stored on the correct device (GPU/CPU).
        """
        cache = {}
        # Tokenize prompts for each aesthetic dimension
        for feature_key in self.positive_prompts:
            cache[feature_key] = {
                # Tokenize positive prompts and move to device
                "pos": clip.tokenize(self.positive_prompts[feature_key]).to(self.device),
                # Tokenize negative prompts and move to device
                "neg": clip.tokenize(self.negative_prompts[feature_key]).to(self.device),
            }
        return cache

    @torch.no_grad()
    def _build_text_feature_cache(self) -> dict:
        """
        Encode all tokenized prompts into normalized text embeddings.
        
        Uses the CLIP text encoder to convert pre-tokenized prompts into feature
        vectors. Features are normalized to unit length (L2 norm = 1) to enable
        efficient cosine similarity computation during scoring.
        
        Returns:
            dict: Nested dictionary mapping aesthetic dimension -> {"pos": features, "neg": features}.
                 Features have shape (num_prompts, 512/768) and L2 norm = 1.
        """
        cache = {}
        # Encode text for each aesthetic dimension
        for feature_key, toks in self._token_cache.items():
            # Encode positive prompt tokens to feature vectors
            p = self.model.encode_text(toks["pos"])
            # Encode negative prompt tokens to feature vectors
            n = self.model.encode_text(toks["neg"])
            # Normalize positive features to unit length
            p = p / p.norm(dim=-1, keepdim=True)
            # Normalize negative features to unit length
            n = n / n.norm(dim=-1, keepdim=True)
            # Store normalized features
            cache[feature_key] = {"pos": p, "neg": n}
        return cache

    @staticmethod
    def _top_k(scores: np.ndarray, prompts: list[str], k: int) -> list[dict]:
        """
        Extract the top-k highest scoring prompts from a set of scores.
        
        Ranks prompts by their similarity scores and returns the top k prompts
        in descending order of score. Handles cases where k > number of prompts
        by capping k to the available number of prompts.
        
        Args:
            scores: Array of similarity scores, one per prompt.
            prompts: List of prompt strings corresponding to scores.
            k: Number of top results to return.
        
        Returns:
            list[dict]: List of dicts with keys "score" (float) and "prompt" (str),
                       sorted in descending order by score.
        """
        # Ensure k doesn't exceed the number of available prompts
        k = min(k, scores.shape[0])
        # Get indices of the k highest scores
        idxs = np.argsort(scores)[-k:][::-1]
        # Build result list with scores and corresponding prompts
        return [{"score": float(scores[i]), "prompt": prompts[i]} for i in idxs]

    @torch.no_grad()
    def score(self, image_input) -> dict:
        """
        Score an image across all aesthetic dimensions.
        
        Computes how well the image aligns with positive and negative prompts
        for each aesthetic dimension. Returns average alignment scores, the delta
        (difference between positive and negative), and the top-k best matching
        prompts for each direction.
        
        Args:
            image_input: Image in any format supported by CLIPModel.encode_image().
        
        Returns:
            dict: Results for each aesthetic dimension with keys:
                - "average_positive": Mean similarity to positive prompts
                - "average_negative": Mean similarity to negative prompts
                - "delta": Difference (positive - negative); higher is better
                - "top_3_positive": Top k positive prompts with scores
                - "top_3_negative": Top k negative prompts with scores
        """
        # Encode the image into a normalized feature vector
        image_feat = self.clip_model.encode_image(image_input)

        results = {}
        # Score the image for each aesthetic dimension
        for feature_key in self.positive_prompts:
            # Get pre-computed text embeddings for this dimension
            p_text = self._text_feat_cache[feature_key]["pos"]
            n_text = self._text_feat_cache[feature_key]["neg"]

            # Compute cosine similarity between image and positive prompts
            # Matrix multiplication @ gives dot product, divide by norms = cosine similarity
            p_scores = (image_feat @ p_text.T).squeeze(0).detach().cpu().numpy()
            # Compute cosine similarity between image and negative prompts
            n_scores = (image_feat @ n_text.T).squeeze(0).detach().cpu().numpy()

            # Calculate average similarity to positive prompts
            avg_p = float(np.mean(p_scores))
            # Calculate average similarity to negative prompts
            avg_n = float(np.mean(n_scores))
            # Delta score: how much better than average negative prompts
            delta = avg_p - avg_n

            # Compile results for this dimension
            results[feature_key] = {
                "average_positive": avg_p,
                "average_negative": avg_n,
                "delta": delta,
                "top_3_positive": self._top_k(p_scores, self.positive_prompts[feature_key], self.top_k),
                "top_3_negative": self._top_k(n_scores, self.negative_prompts[feature_key], self.top_k),
            }

        return results
