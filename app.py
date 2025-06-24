#Import Libraries
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load trained model
@st.cache_resource
def load_trained_model():
    try:
        model = load_model("xray_cnn_model.h5")
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'xray_cnn_model.h5' not found. Please ensure the model file is in the same directory as this app.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

model = load_trained_model()
class_names = ["COVID", "PNEUMONIA", "NORMAL"]
img_size = 128

# Page configuration
st.set_page_config(
    page_title="Chest X-Ray Classifier",
    page_icon="🩻",
    layout="centered"
)

# Title and instructions
st.title("🩻 Chest X-Ray Classifier")
st.write("Upload a chest X-ray image to predict if it's COVID, Pneumonia, or Normal.")
st.write("---")

# Only show upload if model is loaded
if model is not None:
    # Upload image
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear chest X-ray image for analysis"
    )

    if uploaded_file:
        # Show image
        image = Image.open(uploaded_file).convert('L')  # Grayscale
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded X-ray", use_container_width=True)
        
        with col2:
            # Preprocess
            img = image.resize((img_size, img_size))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, img_size, img_size, 1)

            # Predict
            with st.spinner('Analyzing X-ray...'):
                prediction = model.predict(img_array, verbose=0)
                predicted_label = class_names[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

            # Display result with colored boxes
            if predicted_label == "COVID":
                st.error(f"🔴 **Prediction: COVID-19**")
                st.error(f"Confidence: {confidence:.1f}%")
            elif predicted_label == "PNEUMONIA":
                st.warning(f"🟡 **Prediction: PNEUMONIA**")
                st.warning(f"Confidence: {confidence:.1f}%")
            else:
                st.success(f"🟢 **Prediction: NORMAL**")
                st.success(f"Confidence: {confidence:.1f}%")
        
        # Show all confidence scores
        st.write("---")
        st.write("### 📊 Detailed Results")
        
        for i, (label, score) in enumerate(zip(class_names, prediction[0])):
            percentage = score * 100
            # Ensure score is between 0 and 1 for progress bar
            normalized_score = max(0.0, min(1.0, float(score)))
            st.write(f"**{label}:** {percentage:.2f}%")
            st.progress(normalized_score)
        
        # Disclaimer
        st.write("---")
        st.warning("⚠️ **Disclaimer:** This tool is for educational purposes only and should not be used for actual medical diagnosis. Always consult with healthcare professionals.")

else:
    st.info("Please ensure your trained model file 'xray_cnn_model.h5' is in the same directory as this app to proceed.")

# Footer
st.write("---")
st.write("Made with ❤️ using Streamlit and TensorFlow")