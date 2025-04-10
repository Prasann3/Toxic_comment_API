from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os


TOKENIZER_PATH = "tokenizer.pkl"
MODEL_PATH = "toxicity_model.h5"

if not os.path.exists(TOKENIZER_PATH) or not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Tokenizer or Model file not found. Please ensure both are present.")

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

model = load_model(MODEL_PATH)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "https://www.youtube.com"}})

@app.route('/predict', methods=['POST', 'OPTIONS'])
@cross_origin(origins='https://www.youtube.com', methods=['POST', 'OPTIONS'], allow_headers=["Content-Type"])
def predict():
    # Handle preflight request
    if request.method == 'OPTIONS':
        return jsonify({'message': 'CORS preflight success'}), 200

    # 👇 DEBUG LOGGING
    print("🟡 Headers:", dict(request.headers))
    print("🟡 Raw body:", request.data.decode('utf-8'))

    try:
        data = request.get_json(force=True)
        print("🟢 Parsed JSON:", data)
    except Exception as e:
        print("🔴 JSON Parse Error:", str(e))
        return jsonify({"error": "Invalid JSON"}), 400

    comment = data.get("comment", "").strip()
    if not comment:
        return jsonify({"error": "No comment provided"}), 400

    # Preprocess and predict
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    sequence = tokenizer.texts_to_sequences([comment])
    padded = pad_sequences(sequence, maxlen=300)

    try:
        prediction = model.predict(padded)[0]
    except Exception as e:
        return jsonify({"error": "Model prediction failed", "details": str(e)}), 500

    result = {label: float(f"{prob:.3f}") for label, prob in zip(labels, prediction)}

    return jsonify({
        "comment": comment,
        "predictions": result
    })

if __name__ == '__main__':
    app.run(debug=True)
