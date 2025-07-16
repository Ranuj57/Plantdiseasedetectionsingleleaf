import streamlit as st
import tensorflow as tf
import numpy as np

import requests

# Download model at runtime
url = "https://drive.google.com/uc?export=download&id=1vbu7jLi_ksbO2wUKrx-d4jNheTpGGBND"
response = requests.get(url)
with open("finalout2.keras", "wb") as f:
    f.write(response.content)

# Load the model
model = tf.keras.models.load_model("finalout2.keras")

# -------- Model Prediction Function --------
def predict_disease(image_file):
    model = tf.keras.models.load_model("finalout2.keras")
    image = tf.keras.preprocessing.image.load_img(image_file, target_size=(128, 128))
    input_array = tf.keras.preprocessing.image.img_to_array(image)
    input_array = np.expand_dims(input_array, axis=0)
    prediction = model.predict(input_array)
    return np.argmax(prediction)

# -------- Class Labels --------
CLASS_NAMES = [
    'Apple - Scab', 'Apple - Black Rot', 'Apple - Cedar Rust', 'Apple - Healthy',
    'Blueberry - Healthy', 'Cherry - Powdery Mildew', 'Cherry - Healthy',
    'Corn - Gray Leaf Spot', 'Corn - Common Rust', 'Corn - Northern Blight', 'Corn - Healthy',
    'Grape - Black Rot', 'Grape - Black Measles', 'Grape - Leaf Blight', 'Grape - Healthy',
    'Orange - Citrus Greening', 'Peach - Bacterial Spot', 'Peach - Healthy',
    'Pepper - Bacterial Spot', 'Pepper - Healthy', 'Potato - Early Blight',
    'Potato - Late Blight', 'Potato - Healthy', 'Raspberry - Healthy',
    'Soybean - Healthy', 'Squash - Powdery Mildew', 'Strawberry - Leaf Scorch',
    'Strawberry - Healthy', 'Tomato - Bacterial Spot', 'Tomato - Early Blight',
    'Tomato - Late Blight', 'Tomato - Leaf Mold', 'Tomato - Septoria Spot',
    'Tomato - Spider Mites', 'Tomato - Target Spot', 'Tomato - Yellow Leaf Curl Virus',
    'Tomato - Mosaic Virus', 'Tomato - Healthy'
]

# -------- Disease Recognition Page Only --------
st.title("🌿 Plant Disease Recognition")

uploaded_image = st.file_uploader("📤 Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict Disease"):
        st.subheader("Prediction Result")
        predicted_index = predict_disease(uploaded_image)
        st.success(f"The model predicts: **{CLASS_NAMES[predicted_index]}**")
