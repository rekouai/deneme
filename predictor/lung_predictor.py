import torch
import torch.nn as nn
from torchvision import models
from preprocessing.lung import preprocess_from_url

# ✅ Model tanımı (mobilenetv2, 3 sınıf)
def get_model():
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, 3)
    return model

# ✅ Eğitilmiş modeli yükle
model = get_model()
model.load_state_dict(torch.load("models/mobilenet_best_fold1.pt", map_location="cpu"))
model.eval()

# ✅ Tahmin fonksiyonu
def predict(image_url: str):
    x = preprocess_from_url(image_url)  # (1, 3, 224, 224) tensor

    with torch.no_grad():
        out = model(x)                             # (1, 3)
        probs = torch.softmax(out, dim=1).squeeze()  # (3,)
        pred_class = torch.argmax(probs).item()     # sınıf ID'si

    label_map = {0: "Normal", 1: "Benign", 2: "Malign"}

    return {
        "prediction": round(probs[pred_class].item(), 4),
        "label": label_map[pred_class],
        "class_id": pred_class,
        "all_scores": [round(p.item(), 4) for p in probs]
    }
