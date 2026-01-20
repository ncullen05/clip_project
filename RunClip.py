import torch
import clip
from PIL import Image
import numpy as np
from Lighting import lighting
from Background import background
from Perspective import perspective
from Sharpness import sharpness
from Composition import composition

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

all_prompts = {
    "composition": composition,
    "sharpness": sharpness,
    "perspective": perspective,
    "background": background,
    "lighting": lighting,
}

'''
Could we have the different features have a weight of importance as provided by the user
The aesthetics prompts would be separated into different categories: lighting, focus, composition, colour, etc.
In the mobile app, the user selects which categories they care about and provide weights to each prompt
'''

#load the image and prepare it for the model
testImagePath = "images/alfie.jpg"
image = preprocess(Image.open(testImagePath)).unsqueeze(0).to(device)

#List available CLIP models
clip_models = clip.available_models()
print("Available CLIP models:", clip_models)

#convert the labels and prepare it for the model
text = clip.tokenize(all_prompts["background"]).to(device)

with torch.no_grad():
    logitsPerImage, logitsPerText = model(image, text)
    raw_scores = logitsPerImage[0].cpu().numpy()

print("Raw CLIP Scores:", raw_scores)  
sorted_indices = np.argsort(raw_scores)
lowest_number = 3
lowest_indices = sorted_indices[:lowest_number]
print("Your lowest scores were in the following categories:")
for idx in lowest_indices:
    print(f"{all_prompts['background'][idx]}: {raw_scores[idx]}")