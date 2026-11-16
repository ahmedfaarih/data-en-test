import streamlit as st
from transformers import pipeline

@st.cache_resource          # load the model only once
def load_model():
    return pipeline("sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english")

def analyse(text):
    return load_model()(text)[0]

st.title("Sentiment Analyser")
text = st.text_input("Enter some text")
if text:
    result = analyse(text)
    st.write(f"**{result['label']}** (confidence: {result['score']:.2f})")