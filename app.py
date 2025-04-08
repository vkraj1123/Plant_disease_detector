import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
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
    img = img.resize((224, 224))
    img_np = np.array(img)
    if img_np.ndim != 3 or img_np.shape[2] != 3:
        return 0.0
    r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
    green_pixels = (g > r) & (g > b) & (g > 100)
    green_ratio = np.sum(green_pixels) / (224 * 224)
    return green_ratio

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

**Supported Crops**: apple, blueberry, cherry, corn, grape, orange, peach, pepper bell, potato, raspberry, soybean, squash, strawberry, tomato by V
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
        
        if green_ratio < green_threshold:
            st.warning(f"Image appears to have low green content. It may not be a leaf/plant.")
        else:
            pred_class, confidence = predict_image(img)
            if pred_class == "Unknown / Not in database":
                st.warning(f"Low confidence ({confidence:.2f}%). This might not match any known disease.")
            else:
                st.success(f"Predicted Class: **{pred_class}** with **{confidence:.2f}%** confidence.")
            st.progress(int(confidence))

        # Logging values for future feedback/training
        st.sidebar.markdown(f"""
        **Debug Info**
        - Green Ratio: `{green_ratio:.2f}`
        - Threshold Used: `{green_threshold:.2f}`
        """)
