import streamlit as st
import requests

def get_prediction(text):
    url = 'http://localhost:5000/predict'  # Update with your Flask server URL
    data = {'text': text}
    response = requests.post(url, json=data)
    return response.json()['sentiment']

st.title('Sentiment Analysis')
user_input = st.text_area('Enter your text here:')
if st.button('Predict'):
    prediction = get_prediction(user_input)
    st.write(f'Predicted Sentiment: {prediction}')

