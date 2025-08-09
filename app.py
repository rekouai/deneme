from flask import Flask, request, jsonify
from flask_cors import CORS
import os, torch

# küçük CPU'larda thread overhead'i azalt
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# servisler (her biri kendi predictor/preprocess'ini içeriyor)
from services.breast_service import predict_breast
from services.skin_service import predict_skin
from services.lung_service import predict_lung

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://kanser-tani-web.vercel.app"]}})

@app.after_request
def add_cors_headers(resp):
    origin = request.headers.get("Origin")
    if origin:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return resp

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(silent=True) or {}
    cancer_type = data.get("cancer_type")
    if not cancer_type:
        return jsonify({"error": "cancer_type gerekli"}), 400

    try:
        if cancer_type == "breast":
            return predict_breast(data)
        elif cancer_type == "skin":
            return predict_skin(data)
        elif cancer_type == "lung":
            return predict_lung(data)
        else:
            return jsonify({"error": "Geçersiz cancer_type"}), 400
    except Exception as e:
        return jsonify({"error": f"Sunucu hatası: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
