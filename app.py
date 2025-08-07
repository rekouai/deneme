from flask import Flask, request, jsonify
from flask_cors import CORS
import importlib

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        cancer_type = data.get('cancer_type')

        if not cancer_type:
            return jsonify({'error': 'cancer_type alanı zorunlu'}), 400

        # ✅ Meme kanseri: 2 görüntü (CC + MLO)
        if cancer_type == 'breast':
            cc_url = data.get('image_url')
            mlo_url = data.get('image_url_mlo')
            if not cc_url or not mlo_url:
                return jsonify({'error': 'CC ve MLO görüntüleri gerekli'}), 400

            module = importlib.import_module("predictor.breast_ensemble_predictor")
            result = module.predict_with_cc_mlo(cc_url, mlo_url)

        # ✅ Diğer kanser türleri (tek görüntü)
        elif cancer_type in ['skin', 'lung']:
            image_url = data.get('image_url')
            if not image_url:
                return jsonify({'error': 'image_url eksik'}), 400

            module = importlib.import_module(f"predictor.{cancer_type}_predictor")
            result = module.predict(image_url)

        else:
            return jsonify({'error': f'Geçersiz cancer_type: {cancer_type}'}), 400

        return jsonify({
            'cancer_type': cancer_type,
            **result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
