document.addEventListener('DOMContentLoaded', () => {
    const recordButton = document.getElementById('recordButton');
    const stopButton = document.getElementById('stopButton');
    const uploadInput = document.getElementById('uploadInput');
    const audioPlayer = document.getElementById('audioPlayer');
    const emotionResult = document.getElementById('emotionResult');

    let mediaRecorder;
    let chunks = [];

    recordButton.addEventListener('click', startRecording);
    stopButton.addEventListener('click', stopRecording);
    uploadInput.addEventListener('change', handleFileUpload);

    function startRecording() {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.ondataavailable = e => {
                    chunks.push(e.data);
                };
                mediaRecorder.onstop = () => {
                    const blob = new Blob(chunks, { type: 'audio/wav' });
                    const url = URL.createObjectURL(blob);
                    audioPlayer.src = url;
                    uploadAudio(blob);
                    chunks = [];
                };
                mediaRecorder.start();
                recordButton.disabled = true;
                stopButton.disabled = false;
            })
            .catch(err => console.error('Error accessing microphone:', err));
    }

    function stopRecording() {
        if (mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            recordButton.disabled = false;
            stopButton.disabled = true;
        }
    }

    function handleFileUpload() {
        const file = uploadInput.files[0];
        const url = URL.createObjectURL(file);
        audioPlayer.src = url;
        uploadAudio(file);
    }

    function uploadAudio(audioFile) {
        const formData = new FormData();
        formData.append('file', audioFile);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            const emotions = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised'];
            emotionResult.textContent = emotions[data.emotion];
        })
        .catch(err => console.error('Error predicting emotion:', err));
    }
});
