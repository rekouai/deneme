import gc, time, torch
from flask import jsonify
from preprocessors.breast_preprocess import BreastPreprocess
from predictors.breast_predictor import BreastPredictor

# modellerin yolları
BREAST_MLO_PATH = "models/best_xception_seed3407_mlo.pt"
BREAST_CC_PATH  = "models/best_xception_seed123_cc.pt"

# tek sefer yarat, tekrar kullan
_breast_pp = BreastPreprocess()
_breast_predictor = BreastPredictor(BREAST_MLO_PATH, BREAST_CC_PATH, low_ram=True)  # ⬅️ önemli

def predict_breast(data: dict):
    t0 = time.perf_counter()
    cc_url  = data.get("image_url_cc")
    mlo_url = data.get("image_url_mlo")
    if not cc_url or not mlo_url:
        return jsonify({"error": "CC ve MLO görüntü URL'leri gerekli"}), 400

    cc_tensor  = _breast_pp.preprocess(cc_url)
    mlo_tensor = _breast_pp.preprocess(mlo_url)

    pred = _breast_predictor.predict(mlo_tensor, cc_tensor)
    out = {"label": str(pred["label"]), "prediction": float(pred["score"]),
           "latency_ms": round((time.perf_counter()-t0)*1000, 1)}

    # RAM'i derin temizle (low_ram zaten tek tek yüklüyor ama ekstra güvence)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return jsonify(out), 200
