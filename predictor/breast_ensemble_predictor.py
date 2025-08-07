import torch
import torch.nn as nn
from timm import create_model
from torchvision import models
from preprocessing.breast import preprocess_from_url

# 🔧 MODEL TANIMI
def get_model():
    return create_model('xception', pretrained=True, in_chans=1, num_classes=1)

# 🔁 MODELLERİ YÜKLE
model_cc = get_model()
model_cc.load_state_dict(torch.load("models/best_xception_seed123_cc.pt", map_location='cpu'))
model_cc.eval()

model_mlo = get_model()
model_mlo.load_state_dict(torch.load("models/best_xception_seed3407_mlo.pt", map_location='cpu'))
model_mlo.eval()

# 🔍 TAHMİN FONKSİYONU
def predict_with_cc_mlo(cc_url: str, mlo_url: str, threshold: float = 0.44):
    x_cc = preprocess_from_url(cc_url)
    x_mlo = preprocess_from_url(mlo_url)

    with torch.no_grad():
        score_cc = torch.sigmoid(model_cc(x_cc)).item()
        score_mlo = torch.sigmoid(model_mlo(x_mlo)).item()

    ensemble_score = (score_cc + score_mlo) / 2
    label = "Malign" if ensemble_score > threshold else "Benign"

    return {
        "model_cc": round(score_cc, 4),
        "model_mlo": round(score_mlo, 4),
        "prediction": round(ensemble_score, 4),  # Flutter için skor
        "label": label
    }
