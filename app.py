import streamlit as st
import numpy as np
import librosa
import tempfile
from tensorflow.keras.models import load_model

# Load the pre-trained models
def load_emotion_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

lstm_model = load_emotion_model("./emotion_recognition_model_LSTM.keras")
conv1d_model = load_emotion_model("./emotion_recognition_model.keras")

# Function to load an audio file
def load_audio(file_path, sr=22050):
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
        return audio, sample_rate
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None, None

# Function to extract MFCC features
def extract_mfcc(audio, sr, n_mfcc=40):
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    return mfccs

# Function to preprocess the audio file
def preprocess_audio(file):
    audio, sr = load_audio(file)
    if audio is not None:
        mfcc_features = extract_mfcc(audio, sr)
        mfcc_features = np.expand_dims(mfcc_features, axis=0)
        mfcc_features = np.expand_dims(mfcc_features, axis=2)
        return mfcc_features
    return None

# Function to predict emotion
def predict_emotion(model, mfcc_features):
    prediction = model.predict(mfcc_features)
    predicted_emotion = np.argmax(prediction)
    emotion_dict = {
        0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad',
        4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'
    }
    return emotion_dict.get(predicted_emotion, 'unknown')

# Streamlit app layout
st.title("Speech Emotion Detection")

# Model selection
st.header("Select Model")
model_option = st.radio("Choose a model:", ("LSTM", "Conv1D"))

# Select the appropriate model based on user choice
if model_option == "LSTM":
    model = lstm_model
else:
    model = conv1d_model

# Upload audio file
st.header("Upload an Audio File")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name
    
    # Preprocess the uploaded audio file
    mfcc_features = preprocess_audio(temp_file_path)
    if mfcc_features is not None:
        # Make predictions using the selected model
        emotion = predict_emotion(model, mfcc_features)
        st.success(f"Predicted Emotion using {model_option} model: {emotion}")

        # Play the uploaded audio file
        st.audio(uploaded_file, format="audio/wav")