import torch
import clip
from PIL import Image
import numpy as np

from app.prompts import lighting_exposure as le
from app.prompts import contrast as c
from app.prompts import framing_perspective as fp
from app.prompts import visual_focus as vf
from app.prompts import visual_complexity as vc

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval() # Guarantees consistent behaviour. Prevents subtle bugs

positive_prompts = {
    "lighting/exposure": le.p_light_expo,
    "contrast": c.p_contrast,
    "framing/perspective": fp.p_frame_persp,
    "visual focus": vf.p_v_focus,
    "visual complexity": vc.p_v_complexity,
}

negative_prompts = {
    "lighting/exposure": le.n_light_expo,
    "contrast": c.n_contrast,
    "framing/perspective": fp.n_frame_persp,
    "visual focus": vf.n_v_focus,
    "visual complexity": vc.n_v_complexity,
}

# Load the image and prepare it for the model
testImagePath = "images/alfie.jpg"
image = preprocess(Image.open(testImagePath)).unsqueeze(0).to(device)

# Convert the labels and prepare it for the model
token_cache = {}
for feature_key in positive_prompts:
    token_cache[feature_key] = {
        "pos": clip.tokenize(positive_prompts[feature_key]),
        "neg": clip.tokenize(negative_prompts[feature_key])
    }

results = {}
with torch.no_grad():
 # Encode the image once
    image_feat = model.encode_image(image)
    # Scale each embedding so its length is 1, making comparisons fair
    image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True) 

    for feature_key in positive_prompts:
        p_tokens = token_cache[feature_key]["pos"].to(device)
        n_tokens = token_cache[feature_key]["neg"].to(device)

        # Encode text once per set of prompts
        p_text_feat = model.encode_text(p_tokens)
        n_text_feat = model.encode_text(n_tokens)

        p_text_feat = p_text_feat / p_text_feat.norm(dim=-1, keepdim=True)
        n_text_feat = n_text_feat / n_text_feat.norm(dim=-1, keepdim=True)

        # Measure how similar the image is to each prompt using cosine similarity
        p_scores = (image_feat @ p_text_feat.T).squeeze(0).detach().cpu().numpy()
        n_scores = (image_feat @ n_text_feat.T).squeeze(0).detach().cpu().numpy()

        average_p_score = np.mean(p_scores)
        average_p_score = float(average_p_score)  # Convert to a standard Python float
        average_n_score = np.mean(n_scores)
        average_n_score = float(average_n_score)  # Convert to a standard Python float

        delta = average_p_score - average_n_score

        # Get the top 3 positive and negative scores alongside their corresponding prompts
        top_3_p_indices = np.argsort(p_scores)[-3:][::-1]
        top_3_n_indices = np.argsort(n_scores)[-3:][::-1]
        
        top_3_positive = [
            {
                "score": float(p_scores[idx]),
                "prompt": positive_prompts[feature_key][idx]
            }
            for idx in top_3_p_indices
        ]
        
        top_3_negative = [
            {
                "score": float(n_scores[idx]),
                "prompt": negative_prompts[feature_key][idx]
            }
            for idx in top_3_n_indices
        ]

        results[feature_key] = {
            "average_positive": average_p_score,
            "average_negative": average_n_score,
            "delta": delta,
            "positive_scores": p_scores.tolist(),  # Convert numpy array to list for better readability
            "negative_scores": n_scores.tolist(),  # Convert numpy array to list for better readability
            "top_3_positive": top_3_positive,
            "top_3_negative": top_3_negative
        }

print(results)    

'''
print(feature_key)
print("positive:", p_scores)
print("negative:", n_scores)

avg_p_score = np.mean(p_scores)
avg_n_score = np.mean(n_scores)
print(f"Average positive score: {avg_p_score:.4f}")
print(f"Average negative score: {avg_n_score:.4f}")

feature_scores = avg_p_score - avg_n_score
print(f"Feature score (positive - negative): {feature_scores:.4f}")
'''