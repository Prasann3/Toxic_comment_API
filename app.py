from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "https://www.youtube.com"}})

@app.route('/predict', methods=['POST', 'OPTIONS'])
@cross_origin(origin='https://www.youtube.com', methods=['POST', 'OPTIONS'], allow_headers=["Content-Type"])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'CORS preflight success'}), 200

    data = request.get_json()
    comment = data.get("comment")

    if not comment:
        return jsonify({"error": "No comment provided"}), 400

    # Load tokenizer and model
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    model = load_model('toxicity_model.h5')
    sequence = tokenizer.texts_to_sequences([comment])
    padded = pad_sequences(sequence, maxlen=300)
    prediction = model.predict(padded)[0]
    result = {label: float(f"{prob:.3f}") for label, prob in zip(labels, prediction)}

    return jsonify({"comment": comment, "predictions": result})
