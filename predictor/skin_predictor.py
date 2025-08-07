import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from torchvision import models, transforms

# ✅ EfficientNet-B0 modeli (binary classification)
def get_model():
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)  # sigmoid çıkış
    return model

# ✅ Modeli yükle
model = get_model()
model.load_state_dict(torch.load("models/best_augmented_model_val.pth", map_location="cpu"))
model.eval()

# ✅ Preprocessing (resize + normalize)
def preprocess_from_url(image_url: str):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    return transform(img).unsqueeze(0)  # (1, 3, 224, 224)

# ✅ Tahmin fonksiyonu
def predict(image_url: str):
    x = preprocess_from_url(image_url)
    with torch.no_grad():
        out = model(x)  # (1, 1)
        prob = torch.sigmoid(out).item()

    label = "Malign" if prob > 0.5 else "Benign"

    return {
        "prediction": round(prob, 4),  # malign olma ihtimali
        "label": label
    }
