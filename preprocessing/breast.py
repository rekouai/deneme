from PIL import Image, ImageOps
import cv2
import numpy as np
import torchvision.transforms as T

def preprocess_from_url(image_url):
    import requests
    from io import BytesIO

    # Cloudinary'den indir
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert('L')  # grayscale

    # PIL → numpy → cv2 işlemleri
    img_np = np.array(img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img_np)

    _, thresh = cv2.threshold(clahe_img, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found in image.")

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    margin = 30
    x = max(x - margin, 0)
    y = max(y - margin, 0)
    x2 = min(x + w + margin, clahe_img.shape[1])
    y2 = min(y + h + margin, clahe_img.shape[0])
    cropped = clahe_img[y:y2, x:x2]

    # Letterbox resize to 300x300
    desired_size = 300
    old_size = cropped.shape[:2]
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    resized = cv2.resize(cropped, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT, value=0)

    # Tensor'a dönüştür
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    pil_ready = Image.fromarray(padded)
    return transform(pil_ready).unsqueeze(0)
