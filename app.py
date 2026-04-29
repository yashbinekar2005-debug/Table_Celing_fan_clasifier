import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- Configuration ---
MODEL_PATH = 'my_model.h5'
CLASS_NAMES_PATH = 'class_names.txt'
IMG_HEIGHT = 250
IMG_WIDTH = 250

# --- Load Model and Class Names ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_resource
def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        st.error(f"Error: Class names file not found at {CLASS_NAMES_PATH}.")
        st.stop()
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]
    return class_names

model = load_model()
class_names = load_class_names()

# --- Streamlit App ---
st.title("Image Classification App")
st.write("Upload an image and the model will predict its class!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "gif", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img_resized = image.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.success(f"This image most likely belongs to **{predicted_class}** with a **{confidence:.2f}%** confidence.")
else:
    st.info("Awaiting image upload.")
