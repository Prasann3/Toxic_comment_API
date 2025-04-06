from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        # Preflight request for CORS
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Origin'] = 'https://www.youtube.com'
        response.headers['Access-Control-Allow-Methods'] = 'POST'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    # POST request logic
    data = request.get_json()
    comment = data.get("comment")

    if not comment:
        return jsonify({"error": "No comment provided"}), 400

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    model = load_model('toxicity_model.h5')
    sequence = tokenizer.texts_to_sequences([comment])
    padded = pad_sequences(sequence, maxlen=300)
    prediction = model.predict(padded)[0]
    result = {label: float(f"{prob:.3f}") for label, prob in zip(labels, prediction)}

    response = jsonify({"comment": comment, "predictions": result})
    response.headers['Access-Control-Allow-Origin'] = 'https://www.youtube.com'
    return response



if __name__ == '__main__':
    app.run(debug=True)