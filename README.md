
# Music Genre Classification Project

This project is a music genre classifier that uses a Convolutional Neural Network (CNN) to predict the genre of an audio file based on its spectrogram representation.

## Project Structure

* `mu.py`: Contains the code for training the CNN model on spectrogram images.
* `com.py`: Contains the code for loading data, and running the training.
* `model.py`: Contains the code for loading the trained model and making predictions on user-selected audio files.
* `music_genre_classifier.h5`: The saved trained model file (generated after running `mu.py`).
* `audio_data/`: A directory containing audio files organized by genre (e.g., `classical`, `jazz`, `metal`, `pop`).
* `spectrograms/`: A directory containing spectrogram images generated from the audio files.
* `README.md`: This file, providing information about the project.

## Requirements

* Python 3.11 (Recommended)
* TensorFlow
* librosa
* matplotlib
* scikit-learn
* tkinter (for file selection)
* numpy

## Installation

1.  **Clone the repository (if applicable):**

    ```bash
    git clone [repository URL]
    cd [project directory]
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**

    * **Windows:**

        ```bash
        .\.venv\Scripts\activate
        ```

    * **macOS/Linux:**

        ```bash
        source .venv/bin/activate
        ```

4.  **Install the required packages:**

    ```bash
    pip install --upgrade pip
    pip install librosa matplotlib tensorflow scikit-learn tk numpy
    ```

## Usage

1.  **Prepare Audio Data:**
    * Create a directory named `audio_data` in the project's root directory.
    * Inside `audio_data`, create subdirectories for each genre you want to classify (e.g., `classical`, `jazz`, `metal`, `pop`).
    * Place your audio files (.wav or .mp3) into the corresponding genre subdirectories.

2.  **Generate Spectrograms (First Time Only):**
    * If you haven't generated spectrograms yet, run `mu.py` once. This will generate spectrogram images from your audio files and save them in the `spectrograms` directory.
    * Run: `python mu.py`

3.  **Train the Model:**
    * Run the training script: `python com.py`
    * This will train the CNN model using the generated spectrogram images.
    * The trained model will be saved as `music_genre_classifier.h5`.

4.  **Make Predictions:**
    * Run the prediction script: `python model.py`
    * A file dialog will open, allowing you to select an audio file for prediction.
    * The script will then output the predicted genre.

## Notes

* Ensure that your audio files are in a compatible format (.wav or .mp3).
* The training process may take some time, depending on your hardware and dataset size.
* You can adjust the model architecture, hyperparameters, and data augmentation techniques in `mu.py` to improve performance.
* The `spectrogram_dir` variable in the python scripts must be updated to the correct location of your spectrogram images.
* The python version shown in the image is 3.12, but it is recommended to use python 3.11.

