import streamlit as st
import requests

# API Gateway URL (replace with your actual one)
API_URL = "https://teqzjokho5.execute-api.us-east-2.amazonaws.com/Prod/predict"

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("💬 Sentiment Analyzer")
st.write("Enter a sentence and get the sentiment prediction using AWS SageMaker.")

# Text input
text_input = st.text_area("Your text", placeholder="Type something here...")

if st.button("Analyze"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(API_URL, json={"text": text_input})
                result = response.json()["result"][0]
                st.success(f"Prediction: **{result['label']}**")
                st.info(f"Confidence: {round(result['score'] * 100, 2)}%")
            except Exception as e:
                st.error(f"Error: {str(e)}")
