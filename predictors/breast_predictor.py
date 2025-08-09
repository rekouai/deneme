import gc, torch
from timm import create_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BreastPredictor:
    def __init__(self, path_mlo, path_cc, low_ram=True):
        self.path_mlo = path_mlo
        self.path_cc  = path_cc
        self.low_ram  = low_ram
        self.model_mlo = None
        self.model_cc  = None
        if not self.low_ram:
            self.model_mlo = self._load(self.path_mlo)
            self.model_cc  = self._load(self.path_cc)

    def _load(self, path):
        m = create_model('xception', pretrained=False, in_chans=1, num_classes=1)
        state = torch.load(path, map_location=device)
        m.load_state_dict(state, strict=True)
        return m.to(device).eval()

    def _predict_once(self, path, tensor):
        m = self._load(path)
        with torch.inference_mode():
            prob = torch.sigmoid(m(tensor.to(device))).item()
        del m; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return prob

    def predict(self, tensor_mlo, tensor_cc):
        if self.low_ram:
            p1 = self._predict_once(self.path_mlo, tensor_mlo)
            p2 = self._predict_once(self.path_cc,  tensor_cc)
        else:
            with torch.inference_mode():
                p1 = torch.sigmoid(self.model_mlo(tensor_mlo.to(device))).item()
                p2 = torch.sigmoid(self.model_cc(tensor_cc.to(device))).item()
        prob = (p1+p2)/2.0
        label = "Malign" if prob >= 0.44 else "Benign"
        return {"label": label, "score": prob}
