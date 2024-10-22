import streamlit as st
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import numpy as np

model_path = '/Users/apple/Desktop/PG/Summer-24/image-DL/emotion-reconigition/emotion_model'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)

model.eval()

# Emotion labels
file_path = '/Users/apple/Desktop/PG/Summer-24/image-DL/emotion-reconigition/emotions.txt'  # Full path to the file

with open(file_path, 'r') as f:
    emotions = [line.strip() for line in f]


# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

st.title("Emotion Detection from Text")

user_input = st.text_area("Enter text here:")

if st.button("Analyze"):
    inputs = tokenizer(
        user_input,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]
    
    # Get emotions with probability > threshold
    threshold = 0.5
    predicted_emotions = [emotions[i] for i, prob in enumerate(probabilities) if prob > threshold]
    emotion_probs = [prob for prob in probabilities if prob > threshold]
    
    if predicted_emotions:
        st.write("Predicted Emotions:")
        for emotion, prob in zip(predicted_emotions, emotion_probs):
            st.write(f"- {emotion} ({prob:.2f})")
    else:
        st.write("No emotions detected.")
