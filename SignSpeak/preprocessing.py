import json

json_path = "/content/drive/MyDrive/WLASL2000/WLASL_v0.3.json"  # adjust if needed

with open(json_path, 'r') as f:
    metadata = json.load(f)

print("Total classes:", len(metadata))
print("Example sign:", metadata[0])

from google.colab import drive
drive.mount('/content/drive')

import cv2
from tqdm import tqdm

def extract_frames(video_path, max_frames=30, resize=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frame = cv2.resize(frame, resize)
            frames.append(frame)
        count += 1
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames

# Try it on one video
frames = extract_frames(video_label_pairs[0][0])
print("Extracted frames:", len(frames))


import os

video_base_path = '/content/drive/MyDrive/WLASL2000'
video_label_pairs = []

for entry in metadata:
    gloss = entry['gloss']
    for instance in entry['instances']:
        video_id = instance['video_id']
        video_filename = f"{video_id}.mp4"
        video_path = os.path.join(video_base_path, video_filename)

        if os.path.exists(video_path):  # Only include if file exists
            video_label_pairs.append((video_path, gloss))

print(f"Total available videos: {len(video_label_pairs)}")
print("Example:", video_label_pairs[0])


import cv2
import os
import numpy as np
from tqdm import tqdm

def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frame = cv2.resize(frame, (224, 224))  # ResNet50 input size
            frames.append(frame)
    cap.release()

    return np.array(frames)

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model

# Load pretrained ResNet50 (excluding top classification layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_cnn_features(frames):
    preprocessed = preprocess_input(frames)
    features = model.predict(preprocessed, verbose=0)
    features = features.reshape((features.shape[0], -1))  # Flatten spatial dimensions
    return features  # shape: (num_frames, feature_dim)

def prepare_dataset(video_label_list, num_frames=16):
    X, y = [], []
    for video_path, label in tqdm(video_label_list):
        try:
            frames = extract_frames(video_path, num_frames)
            if frames.shape[0] != num_frames:
                continue  # Skip if not enough frames
            features = extract_cnn_features(frames)
            X.append(features)
            y.append(label)
        except:
            print(f"Error processing {video_path}")
            continue
    return np.array(X), np.array(y)


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def encode_labels(labels):
    le = LabelEncoder()
    y_int = le.fit_transform(labels)
    y_cat = to_categorical(y_int)
    return y_cat, le
