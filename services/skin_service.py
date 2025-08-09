import time
from flask import jsonify
from preprocessors.skin_preprocess import SkinPreprocess
from predictors.skin_predictor import SkinPredictor

SKIN_MODEL_PATH = "models/best_augmented_model_val.pth"
_skin_pp = SkinPreprocess()
_skin_pred = SkinPredictor(SKIN_MODEL_PATH)

def predict_skin(data: dict):
    t0 = time.perf_counter()
    img_url = data.get("image_url")
    if not img_url:
        return jsonify({"error": "Görüntü URL'si gerekli"}), 400

    tensor = _skin_pp.preprocess(img_url)
    pred = _skin_pred.predict(tensor)
    out = {"label": str(pred["label"]), "prediction": float(pred["score"]),
           "latency_ms": round((time.perf_counter()-t0)*1000, 1)}
    return jsonify(out), 200
