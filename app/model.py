# app/model.py
"""
CLIP model wrapper for image embedding (Hides the complexity of loading and encoding images with CLIP).

Provides a simple interface to load CLIP and encode images into normalized
feature vectors (a fixed-length list of numbers). 

Supports file paths, bytes, and PIL Images during development.
In production, images will be received as bytes via HTTP.

Automatically selects GPU if available, otherwise falls back to CPU.
"""

import io
import torch
import clip
from PIL import Image

class CLIPModel:

    def __init__(self, model_name: str = "ViT-B/16", device: str | None = None):
        """
        Initialize the CLIP model and preprocessing pipeline.
        
        Args:
            model_name: Name of the CLIP model variant to load (default: ViT-B/16).
                       Other options include ViT-L/14, ViT-B/32, etc.
            device: Device to load the model on ('cuda' or 'cpu'). If None,
                   automatically selects GPU if available, otherwise CPU.
        """
        # Determine the device: use GPU if available, otherwise CPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load the pre-trained CLIP model and get its preprocessing function
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        # Set the model to evaluation mode (disables dropout, batch norm updates, etc.)
        self.model.eval()
        # Store the model name for results
        self.model_name = model_name

    def _to_pil(self, image_input) -> Image.Image:
        """
        Convert various image input formats to PIL Image in RGB format.
            - CLIP requires images to be in RGB format
            - PIL ensures raw image data is model ready
        
        Currently supports three input types for testing purposes. In production,
        images will be received as bytes over HTTP. This method ensures all inputs
        are converted to a consistent RGB PIL Image format.
        
        Args:
            image_input: Image in one of the following formats:
                        - str: Path to image file
                        - bytes or bytearray: Raw image binary data
                        - PIL.Image: Already a PIL Image object
        
        Returns:
            PIL.Image.Image: Image converted to RGB format.
        
        Raises:
            TypeError: If image_input is not one of the supported formats.
        """
        if isinstance(image_input, str):
            # Load image from file path
            pil = Image.open(image_input)
        elif isinstance(image_input, (bytes, bytearray)):
            # Load image from raw binary data (e.g., from HTTP request)
            pil = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            # Already a PIL Image, use directly
            pil = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")
        # Ensure image is in RGB format (handles grayscale, RGBA, etc.)
        return pil.convert("RGB")

    @torch.no_grad()
    def encode_image(self, image_input) -> torch.Tensor:
        """
        Encode an image into a normalized CLIP feature vector.
        
        Converts the input image to a normalized embedding vector using the CLIP
        image encoder. The resulting vector is normalized to unit length (L2 norm = 1),
        making it suitable for cosine similarity comparisons with text embeddings.
            - This ensures a fair comparison of content, not scale.
        
        Args:
            image_input: Image in any of the formats supported by _to_pil().
        
        Returns:
            torch.Tensor: A normalized feature vector of shape (1, 512) for ViT-B/16. 
                         Values are in range [-1, 1] and have L2 norm = 1. 
        """
        # Convert input to PIL Image in RGB format using the helper method
        pil = self._to_pil(image_input)
        # Apply CLIP preprocessing (resize, normalize) and add batch dimension
        image_tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
        # Extract image features from the CLIP encoder
        feat = self.model.encode_image(image_tensor)
        # Normalize features to unit length for cosine similarity computation
        return feat / feat.norm(dim=-1, keepdim=True)