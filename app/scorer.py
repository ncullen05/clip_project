# app/scorer.py
import torch
import clip
import numpy as np

from app.mapping import delta_to_score

class ClipAestheticsScorer:

    def __init__(self, clip_model, positive_prompts: dict, negative_prompts: dict, top_k: int = 3):
        """
        Initialize the aesthetics scorer with CLIP model and prompt sets.
        
        Builds persistent caches (stored data) for tokenized prompts (text prompts 
        converted into numerical form the model can understand) and their encoded features
        during initialization. This one-time setup cost is amortized over many
        scoring calls, making subsequent image evaluations faster.
        
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

        # Token cache: stores the tokenized versions of all prompt strings
        self._token_cache = self._prepare_prompt_tokens()
        # Feature cache: storing the model’s internal representation of images and text as vectors.
        self._text_feat_cache = self._encode_prompt_features()

    def _prepare_prompt_tokens(self) -> dict:
        """
        Pre-tokenize (convert text prompts to numerical forms - tokens) and cache prompts for reuse.
        
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

    @torch.no_grad() # Lets the model know we are not training
    def _encode_prompt_features(self) -> dict:
        """
        The helper method, _prepare_prompt_tokens, returns the tokenized prompts.
        This method builds on that by feeding the tokenized prompts through the CLIP text encoder to get their feature vectors.
        These vectors are then rescaled to a have a length of 1.
        The processed results are then stored in a cache.
        
        Returns:
            dict: Nested dictionary mapping aesthetic dimension -> {"pos": features, "neg": features}.
        """
        cache = {} # Empty container for results
        for feature_key, toks in self._token_cache.items(): # Loop over each aesthetic feature:
            # Feed prompts to CLIP text encoder to get feature vectors
            p = self.model.encode_text(toks["pos"])
            n = self.model.encode_text(toks["neg"])

            # Normalize features to unit length
            p = p / p.norm(dim=-1, keepdim=True)
            n = n / n.norm(dim=-1, keepdim=True)

            # Store normalized features
            cache[feature_key] = {"pos": p, "neg": n}
        return cache

    @staticmethod
    def _top_k(scores: np.ndarray, prompts: list[str], k: int) -> list[dict]:
        """
        It finds the top-k best-matching prompts for an image and returns:
            - the prompt text
            - its similarity score
        
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
        Compute the aesthetic scores
        
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
        # Score the image for each aesthetic feature
        for feature_key in self.positive_prompts:
            # Load the pre-computed embeddings for the prompts
            p_text = self._text_feat_cache[feature_key]["pos"]
            n_text = self._text_feat_cache[feature_key]["neg"]

            # Produce a list of similarity scores between the image and prompts
            p_scores = (image_feat @ p_text.T).squeeze(0).detach().cpu().numpy()
            n_scores = (image_feat @ n_text.T).squeeze(0).detach().cpu().numpy()

            # Calculate average similarity to prompts
            avg_p = float(np.mean(p_scores))
            avg_n = float(np.mean(n_scores))

            # Delta score: how much better than average negative prompts
            delta = avg_p - avg_n
            score_0_10 = delta_to_score(delta, low_delta=-0.10, high_delta=0.10) # Placeholder constants 

            # Compile results for this aesthetic feature
            results[feature_key] = {
                "score_0_10": score_0_10,
                "average_positive": avg_p,
                "average_negative": avg_n,
                "delta": delta,
                "top_3_positive": self._top_k(p_scores, self.positive_prompts[feature_key], self.top_k),
                "top_3_negative": self._top_k(n_scores, self.negative_prompts[feature_key], self.top_k),
            }

        return results