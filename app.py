import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Skin Cancer Detection", page_icon="🩹")

@st.cache_resource
def load_artifacts():
    # ============================================================
    # ⚠️ EDIT THIS LINE: put your Hugging Face repo name below
    #    (find it in your browser address bar, e.g. "johndoe/skin-cancer-model")
    # ============================================================
    HF_REPO = "YOUR_USERNAME/YOUR_REPO"

    # Download the model from Hugging Face (cached after first run)
    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="best_skin_cancer_model.keras"
    )
    model = tf.keras.models.load_model(model_path)

    with open("class_names.json") as f:
        raw = json.load(f)

    # class_names.json may be a LIST  ["benign", "malignant"]
    # or a DICT   {"0": "benign", "1": "malignant"}.
    # Convert either format into a list ordered by class index.
    if isinstance(raw, dict):
        class_names = [raw[str(i)] for i in sorted(raw.keys(), key=lambda k: int(k))]
    else:
        class_names = list(raw)

    if len(class_names) < 2:
        raise ValueError("class_names.json must contain at least 2 class names.")
    return model, class_names

model, class_names = load_artifacts()
IMG_SIZE = (224, 224)

st.title("Skin Cancer Detection (Benign vs Malignant)")
st.caption("Educational demo only — NOT a medical diagnostic tool. Always consult a dermatologist.")

uploaded = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    img = image.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob = float(model.predict(arr, verbose=0).ravel()[0])
    pred_idx = int(prob >= 0.5)

    if pred_idx >= len(class_names):
        st.error(f"Model predicted class index {pred_idx}, but class_names.json only has {len(class_names)} entries.")
        st.stop()

    pred_label = class_names[pred_idx]
    confidence = prob if pred_idx == 1 else 1 - prob

    st.subheader(f"Prediction: **{str(pred_label).upper()}**")
    st.write(f"Confidence: {confidence * 100:.1f}%")
    st.progress(min(max(confidence, 0.0), 1.0))
