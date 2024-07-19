import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from pydub import AudioSegment
from tempfile import NamedTemporaryFile

# Load your trained Speech Emotion Detection model
def load_emotion_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None
    


# Load your trained LSTM model
model = load_emotion_model('emotion_recognition_model.keras')
def convert_mp3_to_wav(mp3_file):
    audio = AudioSegment.from_mp3(mp3_file)
    wav_file = NamedTemporaryFile(delete=False, suffix='.wav')
    audio.export(wav_file.name, format='wav')
    return wav_file.name

# Define emotion labels
emotion_labels = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry','Fearful','Disgust','Surprised']
emotion_emojis = {
    'Neutral': '😐',
    'Calm': '😌',
    'Happy': '😊',
    'Sad': '😢',
    'Angry': '😠',
    'Fearful': '😨',
    'Disgust': '🤢',
}
emotion_suggestions = {
    'Neutral': ["Take a break", "Go for a walk", "Read a book"],
    'Calm': ["Meditate", "Listen to soothing music", "Practice deep breathing"],
    'Happy': ["Celebrate!", "Share your happiness with others", "Listen to upbeat music"],
    'Sad': ["Talk to a friend", "Listen to uplifting music", "Watch a comedy show"],
    'Angry': ["Take deep breaths", "Exercise", "Write down your thoughts"],
    'Fearful': ["Practice mindfulness", "Talk to a loved one", "Listen to calming music"],
    'Disgust': ["Take a moment to relax", "Engage in a favorite hobby", "Watch a funny video"],
    'Surprised': ["Share the surprise with someone", "Take a moment to process", "Reflect on the experience"]
}
def predict_emotion(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file,sr=22050)
    
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    mfccs = mfccs[np.newaxis, ..., np.newaxis]
    
    # Predict emotion
    prediction = model.predict(mfccs)
    predicted_label = emotion_labels[np.argmax(prediction)]
    
    
    return predicted_label
def main():
    st.title('Speech Emotion Detector')
    st.sidebar.title('Choose your option')
    
    choice = st.sidebar.radio("Select an option", ["Upload Audio","Record Audio"])

    
    if choice=="Upload Audio":
     uploaded_file = st.file_uploader("Choose an audio file (.mp3 or .wav)", type=["mp3", "wav"])
     if uploaded_file is not None:
        st.audio(uploaded_file, format='audio')

        # Check file type and convert if MP3
        if uploaded_file.type == 'audio/mp3':
            wav_file = convert_mp3_to_wav(uploaded_file)
            st.audio(wav_file, format='audio')

            prediction = predict_emotion(wav_file)

        else:
            prediction = predict_emotion(uploaded_file)
        emotion_emoji = emotion_emojis[prediction]
        st.markdown(f'{emotion_emoji}  {prediction}', unsafe_allow_html=True)    
       
if __name__ == '__main__':
    main()
