# Voice Emotion Detection

This project is a Streamlit web application that performs speech emotion detection using deep learning models. It allows users to upload audio files and predicts the emotion expressed in the speech using either LSTM or Conv1D neural network models.

## Features

- Upload WAV or MP3 audio files for emotion detection
- Choose between LSTM and Conv1D models for prediction
- Real-time emotion prediction
- Audio playback of uploaded files
- User-friendly interface built with Streamlit

## Technologies Used

- Python
- Streamlit
- TensorFlow / Keras
- Librosa (for audio processing)
- NumPy

## Setup and Installation

1. Clone this repository:
   ```
   git clone https://github.com/SSUR10/Voice-Emotion-Detection.git
   cd Voice-Emotion-Detection
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have the pre-trained model files in the project directory:
   - `emotion_recognition_model_LSTM.keras`
   - `emotion_recognition_model_Conv1D.keras`

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to the URL displayed in the terminal.

3. Select the model you want to use (LSTM or Conv1D).

4. Upload an audio file (WAV or MP3 format).

5. The app will process the audio and display the predicted emotion.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
