import torch
import clip
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

#These are the aesthetic prompts to classify the image
#They will be transferred into an external file to improve flexibility & scalability
aesthetic_prompts = [
    "This photo is aesthetically pleasing.", 
    "The lighting is too dark.",
    "The lighting is overexposed.",
    "The background is distracting.",
    "The image is blurry.",
    "The composition is balanced.",
    "The colors are vibrant and appealing.",
]

#load the image and prepare it for the model
testImagePath = "view.jpg"
image = preprocess(Image.open(testImagePath)).unsqueeze(0).to(device)

#convert the labels and prepare it for the model
text = clip.tokenize(aesthetic_prompts).to(device)

with torch.no_grad():
    logitsPerImage, logitsPerText = model(image, text)
    raw_scores = logitsPerImage[0].cpu().numpy()

print("Raw CLIP Scores:", raw_scores)  
sorted_indices = np.argsort(raw_scores)
lowest_number = 3
lowest_indices = sorted_indices[:lowest_number]
print("Your lowest scores were in the following categories:")
for idx in lowest_indices:
    print(f"{aesthetic_prompts[idx]}: {raw_scores[idx]}")