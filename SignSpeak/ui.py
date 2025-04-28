import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import tempfile
from PIL import Image

# Load pre-trained ResNet model
model = ResNet50(weights='imagenet')

# Streamlit UI
st.title('Sign Language Word Prediction')
st.write('Upload a sign language video to get word predictions using ResNet50.')

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.video(uploaded_file)

    # Button to start prediction with a unique key
    if st.button("Start Prediction", key="start_prediction"):
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()  # Placeholder for displaying frames

        frame_number = 0  # To ensure unique button key for each frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to fit ResNet input size
            img = cv2.resize(frame, (224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Make prediction
            predictions = model.predict(img_array)
            decoded_predictions = decode_predictions(predictions, top=3)[0]

            # Display the frame and predictions
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", caption="Video Frame")

            st.subheader("Top 3 Predictions for Current Frame:")
            for i, (imagenet_id, label, prob) in enumerate(decoded_predictions):
                st.write(f"{i+1}. {label}: {prob*100:.2f}%")

            # Add a unique key to the stop button to avoid duplicate ID issues
            stop_button_key = f"stop_button_{frame_number}"
            if st.button('Stop', key=stop_button_key):
                break

            frame_number += 1  # Increment frame number to ensure unique key

        cap.release()