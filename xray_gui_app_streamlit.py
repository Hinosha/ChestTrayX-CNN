#Import Libraries
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("xray_cnn_model.h5")
class_names = ["COVID", "PNEUMONIA", "NORMAL"]
img_size = 128

# Title and instructions
st.title("🩻 Chest X-Ray Classifier")
st.write("Upload a chest X-ray image to predict if it's COVID, Pneumonia, or Normal.")

# Upload image
uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show image
    image = Image.open(uploaded_file).convert('L')  # Grayscale
    st.image(image, caption="Uploaded X-ray", width=300)

    # Preprocess
    img = image.resize((img_size, img_size))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, img_size, img_size, 1)

    # Predict
    prediction = model.predict(img_array)
    predicted_label = class_names[np.argmax(prediction)]

    # Display result
    st.success(f"### 🧠 Prediction: **{predicted_label}**")
    st.write("Confidence scores:", {label: f"{score:.2%}" for label, score in zip(class_names, prediction[0])})
