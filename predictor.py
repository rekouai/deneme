import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm import create_model
from torchvision import models
import cv2
from PIL import Image
import requests
from io import BytesIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================== 🟣 BREAST ===========================
class BreastPreprocess:
    def __init__(self, image_size=300):
        self.image_size = image_size
        self.transform = A.Compose([
            A.Normalize(mean=0, std=1, max_pixel_value=255.0),
            ToTensorV2()
        ])

    def preprocess(self, url):
        response = requests.get(url)
        img_arr = np.asarray(Image.open(BytesIO(response.content)))
        if img_arr.ndim == 2:
            gray = img_arr.astype(np.uint8)
        else:
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)

        _, thresh = cv2.threshold(clahe_img, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            cropped = clahe_img
        else:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            margin = 30
            x = max(x - margin, 0)
            y = max(y - margin, 0)
            x2 = min(x + w + margin, gray.shape[1])
            y2 = min(y + h + margin, gray.shape[0])
            cropped = clahe_img[y:y2, x:x2]

        old_size = cropped.shape[:2]
        ratio = float(self.image_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        resized = cv2.resize(cropped, (new_size[1], new_size[0]))
        delta_w = self.image_size - new_size[1]
        delta_h = self.image_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - delta_h // 2
        left, right = delta_w // 2, delta_w - delta_w // 2
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        padded = np.expand_dims(padded, axis=-1)
        tensor = self.transform(image=padded)["image"].unsqueeze(0)
        return tensor


class BreastPredictor:
    def __init__(self, model_path_mlo, model_path_cc):
        self.model_mlo = self._load_model(model_path_mlo)
        self.model_cc = self._load_model(model_path_cc)

    def _load_model(self, path):
        model = create_model('xception', pretrained=False, in_chans=1, num_classes=1)
        model.load_state_dict(torch.load(path, map_location=device))
        return model.to(device).eval()
    
    def predict(self, tensor_mlo, tensor_cc):
        with torch.no_grad():
            p1 = torch.sigmoid(self.model_mlo(tensor_mlo.to(device))).item()
            p2 = torch.sigmoid(self.model_cc(tensor_cc.to(device))).item()
            prob = (p1 + p2) / 2
            label_idx = int(prob >= 0.44)
            label = "Malign" if label_idx == 1 else "Benign"
            return {"label": label, "score": prob}

# =========================== 🟢 SKIN ===========================
class SkinPreprocess:
    def __init__(self):
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])

    def preprocess(self, url):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return self.transform(img).unsqueeze(0)


class SkinPredictor:
    def __init__(self, model_path):
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = model.to(device).eval()

    def predict(self, tensor_img):
        with torch.no_grad():
            prob = torch.sigmoid(self.model(tensor_img.to(device))).item()
            label_idx = int(prob >= 0.5)
            label = "Malign" if label_idx == 1 else "Benign"
            return {"label": label, "score": prob}


# =========================== 🔵 LUNG ===========================
class LungPreprocess:
    def __init__(self):
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

    def preprocess(self, url):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return self.transform(img).unsqueeze(0)


class LungPredictor:
    def __init__(self, model_path):
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.last_channel, 3)
        model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = model.to(device).eval()

    def predict(self, tensor_img):
        with torch.no_grad():
            out = self.model(tensor_img.to(device))
            prob = torch.softmax(out, dim=1).squeeze().cpu().numpy()
            label_idx = int(np.argmax(prob))
            class_names = ["Normal", "Benign", "Malign"]
            label = class_names[label_idx]
            return {"label": label, "score": float(np.max(prob)), "all_scores": prob.tolist()}
