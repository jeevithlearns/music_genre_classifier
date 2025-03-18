# train_model.py (Training Script)
import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def create_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, duration=30)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, ax=ax)
    fig.savefig(output_path)
    plt.close(fig)

def generate_spectrograms(audio_dir, spectrogram_dir, genres):
    for genre in genres:
        genre_dir = os.path.join(audio_dir, genre)
        output_genre_dir = os.path.join(spectrogram_dir, genre)
        os.makedirs(output_genre_dir, exist_ok=True)
        for filename in os.listdir(genre_dir):
            if filename.endswith('.wav') or filename.endswith('.mp3'):
                audio_path = os.path.join(genre_dir, filename)
                spectrogram_path = os.path.join(output_genre_dir, filename.replace('.wav', '.png').replace('.mp3', '.png'))
                create_spectrogram(audio_path, spectrogram_path)

def load_data(spectrogram_dir, genres):
    images = []
    labels = []
    for genre in genres:
        genre_dir = os.path.join(spectrogram_dir, genre)
        for filename in os.listdir(genre_dir):
            if filename.endswith('.png'):
                img_path = os.path.join(genre_dir, filename)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 640), color_mode='rgb')
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(genre)
    return np.array(images), np.array(labels)

def build_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    return model

if __name__ == "__main__":
    audio_dir = "audio_data"
    spectrogram_dir = r"C:\Users\harsh\OneDrive\Desktop\mgc\Data\images_original" # Ensure correct path
    genres = ["classical", "jazz", "metal", "pop"]

    X, y = load_data(spectrogram_dir, genres)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_one_hot = tf.keras.utils.to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    input_shape = X_train.shape[1:]
    num_classes = len(genres)
    model = build_model(input_shape, num_classes)
    model = train_model(model, X_train, y_train, X_val, y_val)

    # Save the trained model
    model.save("music_genre_classifier.keras")
    print("Model trained and saved as music_genre_classifier.keras")