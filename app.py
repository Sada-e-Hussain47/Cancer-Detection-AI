import json
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🩹"
)

@st.cache_resource
def load_artifacts():

    # Download model from Hugging Face
    model_path = hf_hub_download(
        repo_id="Sada-e-hussain/skin-cancer-detection",
        filename="best_skin_cancer_model.keras",
        repo_type="model"
    )

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Load class names from GitHub
    with open("class_names.json") as f:
        class_names = json.load(f)

    return model, class_names


model, class_names = load_artifacts()

IMG_SIZE = (224, 224)

st.title("Skin Cancer Detection (Benign vs Malignant)")

st.caption(
    "Educational demo only — NOT a medical diagnostic tool. "
    "Always consult a dermatologist."
)

uploaded = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        use_column_width=True
    )

    img = image.resize(IMG_SIZE)

    arr = np.array(img).astype("float32") / 255.0

    arr = np.expand_dims(arr, axis=0)

    prob = float(
        model.predict(arr, verbose=0).ravel()[0]
    )

    pred_idx = int(prob >= 0.5)

    pred_label = class_names[pred_idx]

    confidence = (
        prob if pred_idx == 1
        else 1 - prob
    )

    st.subheader(
        f"Prediction: **{pred_label.upper()}**"
    )

    st.write(
        f"Confidence: {confidence * 100:.1f}%"
    )

    st.progress(
        min(max(confidence, 0.0), 1.0)
    )
