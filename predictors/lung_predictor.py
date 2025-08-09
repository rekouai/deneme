import torch, torch.nn as nn, numpy as np
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LungPredictor:
    def __init__(self, model_path):
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 3)
        model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = model.to(device).eval()

    def predict(self, tensor_img):
        with torch.inference_mode():
            out = self.model(tensor_img.to(device))
            prob = torch.softmax(out, dim=1).squeeze().cpu().numpy()
        idx = int(np.argmax(prob))
        label = ["Normal","Benign","Malign"][idx]
        return {"label": label, "score": float(prob[idx]), "all_scores": prob.tolist()}
