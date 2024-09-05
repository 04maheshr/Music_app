import streamlit as st
import joblib
import numpy as np
import librosa
from pathlib import Path

# Define the relative paths to the model, scaler, and label encoder
BASE_DIR = Path(__file__).resolve().parent  # Get the directory where the script is located
MODEL_DIR = BASE_DIR / '..' / 'model'
MODEL_PATH = MODEL_DIR / 'ensemble_model.joblib'
SCALER_PATH = MODEL_DIR / 'scaler.joblib'
LABEL_ENCODER_PATH = MODEL_DIR / 'label_encoder.joblib'

# Load the model, scaler, and label encoder
@st.cache_resource  # Caches the loaded model to improve performance
def load_resources():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, scaler, label_encoder

model, scaler, label_encoder = load_resources()

# Define feature extraction function
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    spec_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio).T, axis=0)
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo_feature = np.array([tempo])
    return np.concatenate((mfccs, chroma, spec_contrast, zcr, tempo_feature.flatten()))

# Streamlit app UI
st.title("Music Genre Classification")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    # Extract features from uploaded audio file
    st.write("Extracting features...")
    features = extract_features(uploaded_file)
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Predict genre
    predicted_probs = model.predict_proba(features_scaled)[0]
    st.write("Predicted Probabilities:")
    
    # Display predicted probabilities
    for genre, prob in zip(label_encoder.classes_, predicted_probs):
        st.write(f"{genre}: {prob * 100:.2f}%")

    # Determine the genre with the highest probability
    predicted_genre = label_encoder.classes_[np.argmax(predicted_probs)]
    st.write(f"\nPredicted Genre: {predicted_genre}")
