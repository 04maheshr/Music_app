import os
import numpy as np
import librosa
import joblib

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
MODEL_DIR = os.path.join(BASE_DIR, '../model')
DATA_DIR = os.path.join(BASE_DIR, '../data/your_audio_files_here')

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)

    # Extract features
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    spec_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio).T, axis=0)
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo_feature = np.array([tempo])

    # Concatenate all features into one array
    return np.concatenate((mfccs, chroma, spec_contrast, zcr, tempo_feature.flatten()))

# Load and preprocess data
features, genres = [], []
for genre in os.listdir(DATA_DIR):
    genre_path = os.path.join(DATA_DIR, genre)
    for audio in os.listdir(genre_path):
        file_path = os.path.join(genre_path, audio)
        print(f"Processing file: {file_path}")
        features.append(extract_features(file_path))
        genres.append(genre)

X = np.array(features)
y = np.array(genres)

# Encode the genre labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize individual models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(probability=True, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)

# Create an ensemble model using a Voting Classifier
ensemble_model = VotingClassifier(
    estimators=[('rf', rf_model), ('svm', svm_model), ('knn', knn_model)],
    voting='soft'  # 'soft' voting to use predicted probabilities
)

# Train the ensemble model
ensemble_model.fit(X_train_scaled, y_train)

# Evaluate the model
ensemble_accuracy = accuracy_score(y_test, ensemble_model.predict(X_test_scaled))
print(f"Ensemble Model Accuracy: {ensemble_accuracy * 100:.2f}%")

# Save the model, scaler, and label encoder
joblib.dump(ensemble_model, os.path.join(MODEL_DIR, "ensemble_model.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))
print(f"Model, scaler, and label encoder saved in {MODEL_DIR}")
