import time
from flask import jsonify
from preprocessors.lung_preprocess import LungPreprocess
from predictors.lung_predictor import LungPredictor

LUNG_MODEL_PATH = "models/mobilenet_best_fold1.pt"
_lung_pp = LungPreprocess()
_lung_pred = LungPredictor(LUNG_MODEL_PATH)

def predict_lung(data: dict):
    t0 = time.perf_counter()
    img_url = data.get("image_url")
    if not img_url:
        return jsonify({"error": "Görüntü URL'si gerekli"}), 400

    tensor = _lung_pp.preprocess(img_url)
    pred = _lung_pred.predict(tensor)
    out = {"label": str(pred["label"]), "prediction": float(pred["score"]),
           "latency_ms": round((time.perf_counter()-t0)*1000, 1)}
    return jsonify(out), 200
