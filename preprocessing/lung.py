from PIL import Image
import torchvision.transforms as T
import requests
from io import BytesIO

def preprocess_from_url(image_url):
        # ✅ 1. Cloudinary'den resmi indir
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")  # Akciğer modeli renkli çalışır

    # ✅ 2. Eğitimde kullanılan transformları uygula
    transform = T.Compose([
        T.Resize((224, 224)),  # MobileNetV2 giriş boyutu
        T.ToTensor(),          # (H,W,C) → (C,H,W) + [0,1] aralığı
        T.Normalize(           # ImageNet ortalama ve std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ✅ 3. Tensor'a çevir ve batch dimension ekle
    return transform(img).unsqueeze(0)  # (1, 3, 224, 224)
