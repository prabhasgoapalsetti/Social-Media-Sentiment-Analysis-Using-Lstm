from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH       = os.path.join(BASE_DIR, "sentiment_lstm_model.keras")
TOKENIZER_PATH   = os.path.join(BASE_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

MAX_LEN = 100

# Load NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Load model, tokenizer, label encoder
print("Loading model and artifacts...")
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
with open(LABEL_ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

print("Model and artifacts loaded successfully.")

# ────────────────────────────────────────────────
# TEXT CLEANING (must match training)
# ────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# ────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Clean
        cleaned = clean_text(text)

        # Tokenize & pad
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

        # Predict
        pred = model.predict(padded, verbose=0)
        pred_class = np.argmax(pred, axis=1)[0]
        confidence = float(pred[0][pred_class]) * 100

        sentiment = le.inverse_transform([pred_class])[0]

        return jsonify({
            'sentiment': sentiment,
            'confidence': round(confidence, 2),
            'original_text': text,
            'cleaned_text': cleaned
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    # For production → remove debug=True and use proper server (gunicorn/waitress)    "C:/Users/prabh/Downloads/prabhas project/.venv/Scripts/python.exe" "c:/Users/prabh/Downloads/prabhas project/archive (1)/app.py"    "C:/Users/prabh/Downloads/prabhas project/.venv/Scripts/python.exe" "c:/Users/prabh/Downloads/prabhas project/archive (1)/app.py"
