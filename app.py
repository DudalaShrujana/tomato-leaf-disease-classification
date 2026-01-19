import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# -----------------------------
# SAFE MODEL LOADING
# -----------------------------
MODEL_PATH = os.path.join(os.getcwd(), "tomato_model.h5")

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file NOT FOUND inside container!")
    st.write("Available files:", os.listdir())
    st.stop()

model = load_model(MODEL_PATH)

# -----------------------------
# Class names (exact order as training)
# -----------------------------
class_names = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Healthy"
]

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Tomato Disease Detection", page_icon="🍅")

st.title("🍅 Tomato Plant Disease Detection")
st.write("Upload a tomato leaf image — AI will tell the EXACT disease name.")

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ---- PREPROCESS ----
    img = img.convert("RGB")
    img = img.resize((224, 224))

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ---- PREDICTION ----
    pred = model.predict(img_array)[0]

    pred_class_index = np.argmax(pred)
    pred_class_name = class_names[pred_class_index]
    confidence = float(pred[pred_class_index]) * 100

    # ---- SHOW ONLY EXACT NAME ----
    st.subheader("🧪 Result")

    if confidence < 60:
        st.warning("⚠ Image unclear — please upload a clearer tomato leaf photo.")
    else:
        if pred_class_name == "Healthy":
            st.success("✅ Leaf is HEALTHY — No disease detected")
        else:
            st.error(f"🚨 Disease Detected: {pred_class_name}")

        st.info(f"🎯 Confidence: {confidence:.2f}%")
