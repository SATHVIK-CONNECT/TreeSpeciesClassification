import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import pandas as pd

# Load only the fine-tuned efficient model
@st.cache_resource
def load_model_efficient():
    model = load_model("fine_tuned_efficientnetb0.h5")
    return model

model = load_model_efficient()

# Define image size expected by the model
IMG_SIZE = 224

# Load class names from the dataset folder structure
DATASET_PATH = "Tree_Species_Dataset"
class_names = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])

def preprocess_image(image: Image.Image) -> np.ndarray:
    # Resize and normalize the image
    image = ImageOps.fit(image, (IMG_SIZE, IMG_SIZE))
    image_array = np.asarray(image)
    if image_array.shape[2] == 4:
        # Convert RGBA to RGB
        image_array = image_array[..., :3]
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def main():
    st.title("Tree Species Identification")
    st.write("Upload an image of a plant to identify its species.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        input_array = preprocess_image(image)
        
        # Get prediction from the efficient model
        with st.spinner('Making prediction with the fine-tuned EfficientNet model...'):
            pred = model.predict(input_array)
        
        pred_index = np.argmax(pred)
        pred_confidence = pred[0][pred_index]
        
        # Display result
        st.subheader("Model Prediction")
        st.write(f"Prediction: **{class_names[pred_index]}**")
        st.write(f"Confidence: {pred_confidence:.2f}")

if __name__ == "__main__":
    main()
