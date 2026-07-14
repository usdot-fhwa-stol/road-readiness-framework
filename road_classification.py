from pathlib import Path

import torch
from PIL import Image
from transformers import pipeline


MODEL_ID = "google/siglip2-base-patch16-224"

classifier = pipeline(
    task="zero-shot-image-classification",
    model=MODEL_ID,
    device=0 if torch.cuda.is_available() else -1,
)

image_path = Path("/shared/data/bdd100k/images/test/df1ab122-98fcd29e.jpg")
if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

image = Image.open(image_path).convert("RGB")

candidate_labels = [
    "a forward-facing dashcam image of a road entering or circulating around a roundabout",
    "a forward-facing dashcam image of a curved road",
    "a forward-facing dashcam image of a straight road",
]

predictions = classifier(
    image,
    candidate_labels=candidate_labels,
)

for prediction in predictions:
    print(f"{prediction['label']}: {prediction['score']:.4f}")
