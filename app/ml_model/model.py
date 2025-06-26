import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Model Definition ===
def build_model(input_shape=(None, 64, 64, 3)):
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

# === Training Script (Optional) ===
def train_model(data_dir, categories, output_model_path='violence_detection_model.h5'):
    data = load_video_data(data_dir, categories)
    X, y = prepare_data(data)
    
    model = build_model()
    model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X, y, epochs=10, batch_size=1)
    model.save(output_model_path)
    print(f"✅ Model trained and saved to {output_model_path}")

# === Load Video Data ===
def load_video_data(data_dir, categories):
    data = []
    for category in categories:
        path = os.path.join(data_dir, category)
        label = categories.index(category)
        for video in os.listdir(path):
            video_path = os.path.join(path, video)
            frames = extract_frames(video_path)
            if frames:
                data.append([frames, label])
    return data

# === Extract Frames from a Video ===
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64))
        frames.append(frame)
    cap.release()
    return frames

# === Prepare Input Arrays ===
def prepare_data(data):
    X, y = [], []
    for frames, label in data:
        X.append(frames)
        y.append(label)
    X = pad_sequences(X, padding='post', dtype='float32') / 255.0
    y = np.array(y)
    return X, y

# === Run if script is executed directly ===
if __name__ == "__main__":
    categories = ['non_violent', 'violent']
    train_model(data_dir='../../data', categories=categories)
