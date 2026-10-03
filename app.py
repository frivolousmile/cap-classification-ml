import streamlit as st
from pathlib import Path
import cv2, joblib, numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
IMG_SIZE = (32, 32)

st.set_page_config(page_title="Cap Detection ML", page_icon="🧢")
st.title("🧢 Image-Based Cap Detection")
st.write("Upload an image to classify it as **Cap On** or **Cap Off**.")

if not MODEL_PATH.exists():
    st.error("Model not found. Run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)
uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded:
    data = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not read the image.")
        st.stop()

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Input image", use_container_width=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)
    x = (gray.flatten().astype(np.float32) / 255.0).reshape(1, -1)

    pred = int(model.predict(x)[0])
    label = "CAP ON" if pred == 1 else "CAP OFF"
    st.success(f"Prediction: {label}")
