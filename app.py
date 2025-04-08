import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
from PIL import Image
import os
#Load the models
resnet = ResNet50(weights='imagenet')
model = load_model('plant_disease_model.h5')
#Class labels
class_names = [  # 38 classes
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
# Check if image is likely a plant/leaf
def is_leaf_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = resnet.predict(x)
    label = decode_predictions(preds, top=1)[0][0][1]
    return 'leaf' in label.lower() or 'plant' in label.lower()
#preprocess for disease model
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array
# Prediction logic with confidence threshold
def predict_image(img):
    processed = preprocess_image(img)
    prediction = model.predict(processed)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    if confidence < 70:
        return "Unknown / Not in database", confidence
    else:
        return class_names[predicted_class], confidence
# Streamlit UI 
st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("Plant Disease Detection AI")
st.sidebar.title("About")
st.sidebar.markdown("""Upload a leaf image or use your camera to detect the plant disease. **Supported Crops**: apple, blueberry, cherry, corn, grape, orange, peach, pepper bell, potato, rasbery, soybean, squash, strawberry, tomato only""")
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
# Prediction and UI logic
if img is not None:
    st.image(img, caption=f"Uploaded Image", use_container_width=True)
    with st.spinner("Analyzing image......."):
    if not is_leaf_image(img):
        st.error("This image doesn't appear to be a plant or leaf.")
    else:
        pred_class, confidence = predict_image(img)
        if pred_class == "Unknown / Not in database":
            st.error(f"Low confidence ({confidence:.2f}%). This might not match any known disease.")st.progress(int(confidence))
        else:
            st.success(f"Predicted Class: {pred_class} with {confidence:.2f}% confidence")
            st.progress(int(confidence))
