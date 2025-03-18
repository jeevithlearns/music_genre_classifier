# predict.py (Prediction Script)
import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np
import tensorflow as tf

def predict_audio_genre(model, genres):
    root = tk.Tk()
    root.withdraw()

    audio_file_path = filedialog.askopenfilename(title="Select Audio File", filetypes=(("Audio files", "*.wav;*.mp3"), ("all files", "*.*")))

    if not audio_file_path:
        print("No audio file selected.")
        return

    try:
        y, sr = librosa.load(audio_file_path, duration=30)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = np.expand_dims(S_dB, axis=0)
        img = np.expand_dims(img, axis=-1)

        # Convert to 3 channels
        img = np.repeat(img, 3, axis=-1)

        img = tf.image.resize(img, [128, 640])
        img = img / 255.0

        prediction = model.predict(img)
        predicted_genre_index = np.argmax(prediction)
        predicted_genre = genres[predicted_genre_index]

        print(f"Predicted genre: {predicted_genre}")

    except Exception as e:
        print(f"Error processing audio: {e}")

if __name__ == "__main__":
    # Load the trained model
    model = tf.keras.models.load_model("music_genre_classifier.keras")

    # Replace with your genre list
    genres = ["classical", "jazz", "metal", "pop"]

    # Call the prediction function
    predict_audio_genre(model, genres)