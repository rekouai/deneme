import torch, torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SkinPredictor:
    def __init__(self, model_path):
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = model.to(device).eval()

    def predict(self, tensor_img):
        with torch.inference_mode():
            prob = torch.sigmoid(self.model(tensor_img.to(device))).item()
        label = "Malign" if prob >= 0.5 else "Benign"
        return {"label": label, "score": prob}
