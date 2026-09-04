import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

st.set_page_config(page_title="Pothole Detection", layout="centered")

st.title("🛣️ Pothole Detection with YOLOv8")
st.write("Upload an image to detect and segment potholes dynamically.")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weight", "best.pt")
    return YOLO(model_path)

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Detect Potholes"):
        with st.spinner("Running inference..."):
            results = model.predict(source=image, conf=0.25)
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="Detected Potholes", use_container_width=True)
