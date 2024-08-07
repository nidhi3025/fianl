from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

app = Flask(__name__)

# Load the tokenizer and model
tokenizer = Tokenizer(num_words=5000)  # Ensure this matches your training tokenizer
loaded_model = load_model('path_to_your_model.h5')  # Replace with your actual model path

# Custom functions for text preprocessing if needed
def preprocess_text(tweet):
    # Add your preprocessing steps here if necessary
    return tweet

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    
    # Preprocess the text
    text = preprocess_text(text)
    
    # Tokenize and pad sequences
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=30)
    
    # Make prediction
    prediction = loaded_model.predict(padded_sequences)
    sentiment = np.argmax(prediction, axis=1)[0]  # Assuming softmax output
    
    # Map index to sentiment label
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    predicted_sentiment = sentiment_map[sentiment]
    
    return jsonify({'sentiment': predicted_sentiment})

if __name__ == '__main__':
    app.run(debug=True)
