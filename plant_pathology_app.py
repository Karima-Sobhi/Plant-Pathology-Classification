import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
import cv2

# Load the pre-trained model
model = tf.keras.models.load_model('dense.h5')

st.write("""
         # Plant Pathology Classification
         """
         )

st.write("This is an image classification web app to predict diffrent Plant's Disease")

file = st.file_uploader("Please upload an image file to predict", type=["jpg", "png", "jpeg"])

def import_and_predict(image_data, model):
    size = (128, 128)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)) / 255.
    img_reshape = img_resize[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    try:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        
        # Our Class labels
        class_labels = ["Healthy", "Multiple Diseases", "Rust", "Scab"]
        predicted_label = class_labels[np.argmax(prediction)]
        
        # Display the prediction result with larger font
        # Create custom HTML to increase font size
        font_size = "24px"
        prediction_text = f"<div style='font-size: {font_size};'><b>Prediction:</b> {predicted_label}<div style='font-size: {font_size};'><br>Confidence: {prediction[0][np.argmax(prediction)] * 100:.2f}%"
        st.empty().markdown(prediction_text, unsafe_allow_html=True)
        st.write(prediction)
    except Exception as e:
        st.error(f"Error: {str(e)}")
