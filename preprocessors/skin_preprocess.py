import requests
from io import BytesIO
from PIL import Image
import torchvision.transforms as T

class SkinPreprocess:
    def __init__(self):
        self.t = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        self._http = requests.Session()
        self._http.headers.update({"Connection":"keep-alive"})
        self._timeout = 6

    def preprocess(self, url_or_bytes):
        if isinstance(url_or_bytes, (bytes, BytesIO)):
            img = Image.open(BytesIO(url_or_bytes if isinstance(url_or_bytes, bytes) else url_or_bytes.getvalue())).convert("RGB")
        else:
            r = self._http.get(url_or_bytes, timeout=self._timeout); r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
        return self.t(img).unsqueeze(0)
