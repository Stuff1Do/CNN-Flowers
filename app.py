import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# ----------------------------------------------------------------------------
# Settings (must match the training notebook)
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "flower_classifier.keras"
LABELS_PATH = BASE_DIR / "class_names.json"
IMG_SIZE = 128  # same IMG_SIZE used in Assignment 4
TOP_K = 5

st.set_page_config(page_title="Flower Classifier", page_icon="🌸", layout="wide")


# ----------------------------------------------------------------------------
# Model loading (runs once when the app starts, then is cached)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(LABELS_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


# ----------------------------------------------------------------------------
# Preprocessing: same steps used in training
#   RGB -> resize to 128x128 -> pixel values / 255 -> add batch dimension
# (The model's own Rescaling layer then converts 0-1 to -1..1 for MobileNetV2.)
# ----------------------------------------------------------------------------
def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    array = np.asarray(image, dtype="float32") / 255.0
    return np.expand_dims(array, axis=0)


def predict(model, class_names, image: Image.Image, top_k: int = TOP_K):
    probs = model.predict(preprocess(image), verbose=0)[0]
    top_k = min(top_k, len(class_names))
    top_idx = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in top_idx]


def plot_top_k(results):
    names = [name.title() for name, _ in results]
    scores = [score for _, score in results]
    colors = ["#2e9e4f"] + ["#9aa5b1"] * (len(results) - 1)  # top-1 in green

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(names, scores, color=colors)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title(f"Top-{len(results)} predicted classes")
    for bar, score in zip(bars, scores):
        ax.text(
            min(score + 0.02, 0.85),
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1%}",
            va="center",
        )
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Sidebar: title, description, instructions
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🌸 Flower Classifier")
    st.write(
        "This app uses a deep learning model (MobileNetV2 with transfer learning) "
        "trained on the Kaggle Flowers Recognition dataset to identify a flower "
        "as a daisy, dandelion, rose, sunflower, or tulip."
    )
    st.subheader("How to use")
    st.markdown(
        "1. Upload a flower photo (.jpg, .jpeg, or .png).\n"
        "2. Click **Predict**.\n"
        "3. Read the predicted class and the confidence chart."
    )
    st.caption("CMSC 170 – Assignment 5 · UP Tacloban")

# ----------------------------------------------------------------------------
# Main panel
# ----------------------------------------------------------------------------
st.header("Upload an image")

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.error(
        "Model files not found. Put `flower_classifier.keras` and "
        "`class_names.json` in the same folder as `app.py`."
    )
    st.stop()

model, class_names = load_model_and_labels()

uploaded = st.file_uploader(
    "Drag and drop a file here, or browse", type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.info("Upload an image to get started.")
else:
    image = ImageOps.exif_transpose(Image.open(uploaded))
    col_left, col_right = st.columns(2)

    with col_left:
        st.image(image, caption=uploaded.name, use_container_width=True)
        predict_clicked = st.button("Predict", type="primary")

    if predict_clicked:
        with st.spinner("Classifying..."):
            results = predict(model, class_names, image)
        top_name, top_score = results[0]

        with col_right:
            st.subheader("Result")
            st.success(f"Prediction: **{top_name.title()}** ({top_score:.1%} confidence)")
            st.pyplot(plot_top_k(results))