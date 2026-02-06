import torch
import clip
from PIL import Image
import numpy as np

import lighting_exposure_analysis as le
import contrast_analysis as c
import framing_perspective_analysis as fp
import visual_focus_analysis as vf
import visual_complexity_analysis as vc

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

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

#load the image and prepare it for the model
testImagePath = "images/alfie.jpg"
image = preprocess(Image.open(testImagePath)).unsqueeze(0).to(device)

#convert the labels and prepare it for the model
for feature_key in positive_prompts:
    pos_list = positive_prompts[feature_key]
    neg_list = negative_prompts[feature_key]
    
    p_text = clip.tokenize(pos_list).to(device)
    n_text = clip.tokenize(neg_list).to(device)

    with torch.no_grad():
        # positive prompts
        p_logits_img, _ = model(image, p_text)
        p_scores = p_logits_img[0].cpu().numpy()

        # negative prompts
        n_logits_img, _ = model(image, n_text)
        n_scores = n_logits_img[0].cpu().numpy()

    print(feature_key)
    print("positive:", p_scores)
    print("negative:", n_scores)

    avg_p_score = np.mean(p_scores)
    avg_n_score = np.mean(n_scores)
    print(f"Average positive score: {avg_p_score:.4f}")
    print(f"Average negative score: {avg_n_score:.4f}")

    feature_scores = avg_p_score - avg_n_score
    print(f"Feature score (positive - negative): {feature_scores:.4f}")