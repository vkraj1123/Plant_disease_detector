import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
from PIL import Image
import csv
from datetime import datetime
import os

# Create a logging function
def log_prediction(image_name, green_ratio, green_threshold, predicted_class, confidence):
    log_file = "plant_disease_log.csv"
    log_exists = os.path.exists(log_file)

    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not log_exists:
            writer.writerow(["Timestamp", "Image Name", "Green Pixel Ratio", "Green Threshold", "Predicted Class", "Confidence (%)"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_name, green_ratio, green_threshold, predicted_class, f"{confidence:.2f}"])

# page config first
st.set_page_config(page_title="Plant Disease Detector", layout="centered")

# ---------------------------------------------
# Load Plant Disease Model
# ---------------------------------------------
@st.cache_resource
def load_disease_model():
    return load_model('plant_disease_model.h5')

model = load_disease_model()

# ---------------------------------------------
# Class labels
# ---------------------------------------------
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ---------------------------------------------
# Green Pixel Ratio Leaf Detector
# ---------------------------------------------
def get_green_ratio(img):
    img_np = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Define standard green range
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Define dark green range
    lower_dark_green = np.array([35, 40, 20])
    upper_dark_green = np.array([85, 255, 80])
    mask_dark_green = cv2.inRange(hsv, lower_dark_green, upper_dark_green)

    # Combine both masks
    combined_mask = cv2.bitwise_or(mask_green, mask_dark_green)
    green_pixels = cv2.countNonZero(combined_mask)
    total_pixels = img_np.shape[0] * img_np.shape[1]

    green_ratio = green_pixels / total_pixels
    return green_ratio

# ---------------------------------------------
# Leaf Shape Detector
# ---------------------------------------------
def is_leaf_shape(img: Image.Image) -> bool:
    img_np = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_like_contours = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area > 500 and perimeter > 100:
            circularity = 4 * np.pi * (area / (perimeter * perimeter + 1e-5))
            if 0.3 < circularity < 1.2:
                leaf_like_contours += 1
    return leaf_like_contours >= 1

# ---------------------------------------------
# Preprocess image
# ---------------------------------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# ---------------------------------------------
# Predict Disease
# ---------------------------------------------
def predict_image(img):
    processed = preprocess_image(img)
    prediction = model.predict(processed)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    if confidence < 70:
        return "Unknown / Not in database", confidence
    else:
        return class_names[predicted_class], confidence

# ---------------------------------------------
# Streamlit UI
# ---------------------------------------------
st.title("Plant Disease Detection AI")
st.sidebar.title("About")
st.sidebar.markdown("""
Upload a leaf image or use your camera to detect the plant disease.

**Supported Crops**: apple, blueberry, cherry, corn, grape, orange, peach, pepper bell, potato, raspberry, soybean, squash, strawberry, tomato by Vk
""")

option = st.radio("Choose input method:", ('Upload from Gallery', 'Capture from Camera'))
img = None

if option == 'Upload from Gallery':
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)

elif option == 'Capture from Camera':
    picture = st.camera_input("Take a picture")
    if picture is not None:
        img = Image.open(picture)

# ---------------------------------------------
# Threshold Slider for Green Ratio
# ---------------------------------------------
st.sidebar.subheader("Green Detection Sensitivity")
green_threshold = st.sidebar.slider("Min. green pixel ratio", 0.05, 0.8, 0.25, step=0.01)

# ---------------------------------------------
# Main Logic
# ---------------------------------------------
if img is not None:
    st.image(img, caption="Uploaded Image", use_container_width=True)
    with st.spinner("Analyzing image..."):
        green_ratio = get_green_ratio(img)
        st.info(f"Detected Green Pixel Ratio: **{green_ratio:.2f}**")

        if green_ratio < green_threshold or not is_leaf_shape(img):
            st.warning(f"Image does not appear to be a proper leaf. Please upload a clearer leaf image.")
        else:
            pred_class, confidence = predict_image(img)
            log_prediction("uploaded_image.jpg", green_ratio, green_threshold, pred_class, confidence)
            if pred_class == "Unknown / Not in database":
                st.warning(f"Low confidence ({confidence:.2f}%). This might not match any known disease.")
            else:
                st.success(f"Predicted Class: **{pred_class}** with **{confidence:.2f}%** confidence.")
            st.progress(int(confidence))

        st.sidebar.markdown(f"""
        **Debug Info**
        - Green Ratio: `{green_ratio:.2f}`
        - Threshold Used: `{green_threshold:.2f}`
        """)

    if st.sidebar.checkbox("Show detected leaf contours"):
        img_np = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = img_np.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        st.image(contour_img, caption="Detected Contours", channels="RGB")
