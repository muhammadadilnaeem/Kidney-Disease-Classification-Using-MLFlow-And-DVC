import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

# Streamlit page configuration
st.set_page_config(
    page_title="Kidney Disease Classifier",
    page_icon="🫁",
    layout="centered"
)

# Modernized CSS styling for UI/UX
st.markdown(
    """
    <style>
        .main {
            background-color: #f8f9fd;
        }
        .header {
            color: #ffffff;
            text-align: center;
            font-family: 'Verdana', sans-serif;
            background-color: #6c5ce7;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }
        .upload-box {
            text-align: center;
            border: 2px dashed #0984e3;
            padding: 25px;
            margin-top: 20px;
            margin-bottom: 30px;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
        }
        .stButton > button {
            background-color: #00cec9;
            color: #ffffff;
            padding: 15px 30px;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            font-size: 20px;
            font-weight: bold;
            margin-top: 20px;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #009688;
        }
        .button-container {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        .result-box {
            border: 4px solid #2ecc71;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-top: 30px;
            font-size: 20px;
            font-weight: bold;
            color: #2ecc71;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }
        .result-box.disease {
            border-color: #e74c3c;
            color: #e74c3c;
        }
        .instruction {
            text-align: center;
            color: #636e72;
            margin-bottom: 20px;
            font-size: 18px;
            font-family: 'Arial', sans-serif;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Display a heading with improved text
st.markdown('<div class="header"><h1>🫁 Kidney Prediction System 🫁</h1></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="instruction">This tool uses a deep learning model to analyze chest X-ray images and predict if the patient is <b>healthy</b> or may have <b>Tumor</b>. Upload a chest X-ray image to get started.</div>',
    unsafe_allow_html=True
)

# File uploader
st.markdown('<div class="upload-box"><h3>📤 Upload your X-Ray Image below (JPEG, PNG, JPG)</h3></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📂 Select an image file...", type=["png", "jpg", "jpeg"])

# Class for prediction pipeline
class PredictionPipeline:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def preprocess_image(self, image):
        """
        Preprocesses the uploaded image to ensure compatibility with the model.
        - Converts to RGB if necessary.
        - Resizes the image to (224, 224).
        - Converts to a NumPy array with appropriate dimensions.
        """
        # Convert image to RGB if it has an alpha channel or grayscale
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize image to match the model's input shape
        resized_image = image.resize((224, 224))
        
        # Convert the image to a NumPy array
        image_array = img_to_array(resized_image)
        
        # Expand dimensions to match the model's input (1, 224, 224, 3)
        preprocessed_image = np.expand_dims(image_array, axis=0)
        
        # Normalize pixel values (optional, based on your model's training)
        preprocessed_image /= 255.0
        
        return preprocessed_image

    def predict(self, image):
        """
        Predicts the class of the uploaded image.
        - Preprocesses the image.
        - Passes it through the model for prediction.
        - Returns a user-friendly result string.
        """
        # Preprocess the image
        preprocessed_image = self.preprocess_image(image)
        
        # Predict using the loaded model
        result = np.argmax(self.model.predict(preprocessed_image), axis=1)
        
        # Return prediction result
        return "Tumor" if result[0] == 1 else "Normal"

# Load the model only once to save resources
MODEL_PATH = r"artifacts/training/model.h5"
pipeline = PredictionPipeline(MODEL_PATH)

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded X-Ray Image", use_container_width=True)

    # Centered prediction button
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("🔍 Predict"):
        prediction = pipeline.predict(image)
        # Conditional styling for the result box
        if prediction == "Normal":
            st.markdown(
                f"<div class='result-box'>🌟 <b>Prediction:</b> {prediction} - No signs of disease detected. 🌟</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box disease'>⚠️ <b>Prediction:</b> {prediction} - Consult a doctor immediately for further diagnosis. ⚠️</div>",
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)