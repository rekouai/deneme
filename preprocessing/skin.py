from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as T
import requests
from io import BytesIO

# Hair removal (gray üzerinden morphology + inpainting)
def remove_hair(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(1, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    result = cv2.inpaint(img_rgb, mask, 1, cv2.INPAINT_TELEA)
    return result

# CLAHE (LAB color space)
def apply_clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# Ana preprocessing fonksiyonu
def preprocess_from_url(image_url: str):
    # Görüntüyü indir
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img_np = np.array(img)

    # Hair removal
    img_np = remove_hair(img_np)

    # CLAHE
    img_np = apply_clahe(img_np)

    # Resize to 224x224
    img_np = cv2.resize(img_np, (224, 224))

    # Tensor'a dönüştür
    transform = T.Compose([
        T.ToTensor(),  # (H, W, C) → (C, H, W) & [0, 255] → [0.0, 1.0]
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Aynı eğitimdeki gibi
    ])

    tensor = transform(Image.fromarray(img_np))
    return tensor.unsqueeze(0)  # (1, 3, 224, 224)
