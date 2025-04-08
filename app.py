import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load the trained model once
@st.cache_resource
def load_disease_model():
    return load_model('plant_disease_model.h5')

model = load_disease_model()

# Helper function to preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))  # Assuming model expects 224x224 input
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# Predict disease from image
def predict_disease(img):
    preprocessed = preprocess_image(img)
    prediction = model.predict(preprocessed)[0]
    return prediction

# Green ratio analysis
def get_green_ratio(img):
    img_np = np.array(img)
    if img_np.shape[-1] == 4:
        img_np = img_np[..., :3]

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Define range for green color in HSV
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    green_pixels = np.sum(mask > 0)
    total_pixels = img_np.shape[0] * img_np.shape[1]

    green_ratio = green_pixels / total_pixels
    return green_ratio

# Streamlit UI
st.set_page_config(page_title="Krishak Sathi - Plant Doctor", layout="centered")

st.title("Krishak Sathi - Plant Disease Detector")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Leaf Image", use_column_width=True)

        # Prediction
        st.subheader("Disease Prediction")
        predictions = predict_disease(img)
        class_names = ['Apple___Black_rot', 'Apple___healthy', 'Corn_(maize)___Cercospora_leaf_spot',
                       'Corn_(maize)___healthy', 'Grape___Black_rot', 'Potato___Early_blight',
                       'Potato___healthy', 'Tomato___Late_blight', 'Tomato___healthy']  # example classes
        top_prediction = class_names[np.argmax(predictions)]
        st.success(f"Prediction: **{top_prediction}**")

        # Confidence
        st.write("Confidence levels:")
        for i, prob in enumerate(predictions):
            st.write(f"{class_names[i]}: {prob:.2%}")

        # Green Ratio
        st.subheader("Leaf Health Analysis (Green Ratio)")
        green_ratio = get_green_ratio(img)
        st.write(f"Green Pixel Ratio: **{green_ratio:.2%}**")

        if green_ratio < 0.3:
            st.warning("Low green ratio may indicate unhealthy or diseased leaf.")
        elif green_ratio > 0.8:
            st.success("High green ratio — leaf looks healthy!")
        else:
            st.info("Moderate green ratio — partial stress or discoloration possible.")

    except Exception as e:
        st.error("An error occurred while processing the image.")
        st.exception(e)
