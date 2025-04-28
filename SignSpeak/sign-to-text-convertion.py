import kagglehub
risangbaskoro_wlasl_processed_path = kagglehub.dataset_download('risangbaskoro/wlasl-processed')

print('Data source import complete.')

# ====== Install & Import Libraries ======
!pip install -q tensorflow
import os
import cv2
import numpy as np
import json
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# ====== Load metadata and set video paths ======
with open('/kaggle/input/wlasl-processed/WLASL_v0.3.json') as f:
    metadata = json.load(f)

video_base_path = '/kaggle/input/wlasl-processed'
video_label_pairs = []

for entry in metadata:
    gloss = entry['gloss']
    for instance in entry['instances']:
        video_id = instance['video_id']
        video_path = os.path.join(video_base_path, 'videos', f"{video_id}.mp4")
        if os.path.exists(video_path):
            video_label_pairs.append((video_path, gloss))

print(f"✅ Total available videos: {len(video_label_pairs)}")

# ====== CNN Feature Extractor ======
resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

def extract_cnn_features(frames):
    frames = np.array(frames)
    frames = preprocess_input(frames)
    features = resnet.predict(frames, verbose=0)
    return features

# ====== Frame Extractor ======
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
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    cap.release()
    return np.array(frames)

# ====== Dataset Preparation ======
def prepare_dataset(video_label_list, num_frames=16):
    X, y = [], []
    for video_path, label in tqdm(video_label_list):
        try:
            frames = extract_frames(video_path, num_frames)
            if frames.shape[0] != num_frames:
                continue
            features = extract_cnn_features(frames)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"⚠️ Error processing {video_path}: {e}")
            continue
    return np.array(X), np.array(y)

def encode_labels(labels):
    le = LabelEncoder()
    y_int = le.fit_transform(labels)
    y_cat = to_categorical(y_int)
    return y_cat, le

# ====== Extract and Save ======
X, y_raw = prepare_dataset(video_label_pairs)
y, label_encoder = encode_labels(y_raw)

print("✅ Shape of X:", X.shape)
print("✅ Shape of y:", y.shape)

# Save arrays to disk
np.save('/kaggle/working/X_features.npy', X)
np.save('/kaggle/working/y_labels.npy', y)
import pickle
with open('/kaggle/working/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✅ Saved features and labels!")


# First, zip the X_features.npy file
!zip /kaggle/working/X_features.zip /kaggle/working/X_features.npy

# ====== Zip X_features.npy ======
import shutil

# Create a zip file containing only X_features.npy
shutil.make_archive('/kaggle/working/X_features', 'zip', root_dir='/kaggle/working', base_dir='X_features.npy')

print("✅ X_features.npy zipped successfully!")

# ====== Download the zip ======
from IPython.display import FileLink

# Create a clickable download link
FileLink(r'/kaggle/working/X_features.zip')


# ====== Install & Import Libraries ======
!pip install -q keras-tuner tensorflow
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, TimeDistributed
import keras_tuner as kt

# ====== Load Saved Data ======
X = np.load('/kaggle/working/X_features.npy')
y = np.load('/kaggle/working/y_labels.npy')
with open('/kaggle/working/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print("✅ Loaded features and labels")

# ====== Train/Test Split ======
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
input_shape = (X.shape[1], X.shape[2])  # (16, 2048)
num_classes = y.shape[1]

# ====== Define Model Builder for Hyperparameter Tuning ======
def build_lstm_model(hp):
    model = Sequential([
        Bidirectional(LSTM(hp.Int('lstm_units', min_value=128, max_value=512, step=64), return_sequences=True), input_shape=input_shape),
        Dropout(hp.Float('dropout_rate', min_value=0.3, max_value=0.6, step=0.1)),
        LSTM(hp.Int('lstm_units_2', min_value=64, max_value=256, step=64)),
        Dropout(hp.Float('dropout_rate_2', min_value=0.3, max_value=0.6, step=0.1)),
        Dense(hp.Int('dense_units', min_value=128, max_value=512, step=64), activation='relu'),
        Dropout(hp.Float('dropout_rate_3', min_value=0.3, max_value=0.6, step=0.1)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_gru_model(hp):
    model = Sequential([
        Bidirectional(GRU(hp.Int('gru_units', min_value=128, max_value=512, step=64), return_sequences=True), input_shape=input_shape),
        Dropout(hp.Float('dropout_rate', min_value=0.3, max_value=0.6, step=0.1)),
        GRU(hp.Int('gru_units_2', min_value=64, max_value=256, step=64)),
        Dropout(hp.Float('dropout_rate_2', min_value=0.3, max_value=0.6, step=0.1)),
        Dense(hp.Int('dense_units', min_value=128, max_value=512, step=64), activation='relu'),
        Dropout(hp.Float('dropout_rate_3', min_value=0.3, max_value=0.6, step=0.1)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_cnn_lstm_model(hp):
    model = Sequential([
        TimeDistributed(Dense(hp.Int('cnn_units', min_value=128, max_value=512, step=64), activation='relu'), input_shape=input_shape),
        Dropout(hp.Float('dropout_rate', min_value=0.3, max_value=0.6, step=0.1)),
        LSTM(hp.Int('lstm_units', min_value=128, max_value=512, step=64), return_sequences=True),
        Dropout(hp.Float('dropout_rate_2', min_value=0.3, max_value=0.6, step=0.1)),
        LSTM(hp.Int('lstm_units_2', min_value=64, max_value=256, step=64)),
        Dropout(hp.Float('dropout_rate_3', min_value=0.3, max_value=0.6, step=0.1)),
        Dense(hp.Int('dense_units', min_value=128, max_value=512, step=64), activation='relu'),
        Dropout(hp.Float('dropout_rate_4', min_value=0.3, max_value=0.6, step=0.1)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ====== Hyperparameter Tuning (Fixed number of trials = 10) ======
def run_tuning(model_builder, model_name):
    tuner = kt.RandomSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=10,  # Set exactly 10 trials
        executions_per_trial=1,
        directory='/kaggle/working/',
        project_name=f'{model_name}_tuning'
    )
    
    tuner.search(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.summary()
    
    # Save best model
    best_model.save(f'/kaggle/working/{model_name}_best_model.h5')
    
    # Get the best hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"\n✅ Best hyperparameters for {model_name}: {best_hp.values}")
    
    # Print number of trials
    print(f"✅ Total number of trials for {model_name}: {len(tuner.oracle.get_trials())}")
    
    return best_model, best_hp

# ====== Run Tuning for All Models ======
print("\n🚀 Running Hyperparameter Tuning for LSTM...")
best_lstm_model, best_lstm_hp = run_tuning(build_lstm_model, "lstm")

print("\n🚀 Running Hyperparameter Tuning for GRU...")
best_gru_model, best_gru_hp = run_tuning(build_gru_model, "gru")

print("\n🚀 Running Hyperparameter Tuning for CNN-LSTM...")
best_cnn_lstm_model, best_cnn_lstm_hp = run_tuning(build_cnn_lstm_model, "cnn_lstm")

# ====== Evaluate Best Models ======
print("\n✅ LSTM Best Model Validation Accuracy:", best_lstm_model.evaluate(X_val, y_val, verbose=0)[1])
print("\n✅ GRU Best Model Validation Accuracy:", best_gru_model.evaluate(X_val, y_val, verbose=0)[1])
print("\n✅ CNN-LSTM Best Model Validation Accuracy:", best_cnn_lstm_model.evaluate(X_val, y_val, verbose=0)[1])

