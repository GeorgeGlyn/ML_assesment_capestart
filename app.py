from flask import Flask, request, jsonify
import joblib
from sentence_transformers import SentenceTransformer
import threading

app = Flask(__name__)

model = joblib.load('article_classifier.pkl')
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    text_cleaned = cleaning_text(text)
    embedding = sentence_model.encode([text_cleaned])
    prediction = model.predict([embedding])[0]
    return jsonify({'prediction': prediction})

def my_app():
    app.run(debug=True, use_reloader=False)

thread = threading.Thread(target=my_app)
thread.start()