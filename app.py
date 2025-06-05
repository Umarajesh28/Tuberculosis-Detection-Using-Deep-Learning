import streamlit as st
import tensorflow as tf
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import base64
import io
import os
import gdown

# --- CONFIG ---
st.set_page_config(page_title="TB Detection App", layout="centered")

# --- MODEL DOWNLOAD AND LOAD ---
MODEL_PATH = "saved_models/vgg16_tb_model.keras"
MODEL_URL = "https://drive.google.com/uc?id=1XfYMP0rDl9v5Zg5sP7_DBRvPENn2U1_0"  # Your model's Google Drive ID

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("saved_models", exist_ok=True)
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

@st.cache_resource
def load_model_local():
    download_model()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model_local()



class_names = ["Normal", "Tuberculosis"]

def preprocess(img_file):
    img = Image.open(img_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img

def predict(img_file):
    img_array, display_img = preprocess(img_file)
    pred = model.predict(img_array)[0][0]
    label = "Tuberculosis" if pred > 0.5 else "Normal"
    confidence = pred if pred > 0.5 else 1 - pred
    return label, float(confidence), display_img, img_array

def generate_report(label, confidence):
    report = f"Prediction: {label}\nConfidence: {confidence:.2%}\nThank you for using the TB Detection App."
    b64 = base64.b64encode(report.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="tb_report.txt">Download Report</a>'
    return href

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    page = option_menu("TB Detection App", ["Home", "Predict"], icons=["house", "search"], menu_icon="cast", default_index=0)

# --- HOME PAGE ---
if page == "Home":
    st.title("Welcome to the Tuberculosis Detection App 🩺")
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966485.png", width=100)
    st.markdown("""
        This application allows you to detect signs of **Tuberculosis** from chest X-ray images using a deep learning model.

        - Built with VGG16 architecture  
        - Powered by TensorFlow & Streamlit  

        Use the sidebar to navigate to the **Prediction** page.
    """)

# --- PREDICT PAGE ---
elif page == "Predict":
    st.title("Upload a Chest X-ray Image")
    st.subheader("Try a sample X-ray or upload your own")

    # Load sample images
    sample_dir = "sample_images"
    samples = [f for f in os.listdir(sample_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
    sample_choice = st.selectbox("Choose a sample X-ray", ["None"] + samples)

    uploaded_file = st.file_uploader("Or upload your own chest X-ray", type=["jpg", "jpeg", "png"])

    image_to_use = None

    if uploaded_file is not None:
        image_to_use = uploaded_file
        st.info(" Using uploaded image.")
    elif sample_choice != "None":
        sample_path = os.path.join(sample_dir, sample_choice)
        image_to_use = open(sample_path, "rb")
        st.info(f" Using sample image: {sample_choice}")

    if image_to_use is not None:
        label, confidence, display_img, img_array = predict(image_to_use)

        st.image(display_img, caption="Selected Chest X-ray", use_container_width=True)
        st.success(f"### Prediction: **{label}**")
        st.info(f"Confidence: `{confidence:.2%}`")
        st.markdown(generate_report(label, confidence), unsafe_allow_html=True)
