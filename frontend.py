import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Potato Leaf Classifier", page_icon="🥔", layout="centered")

st.title("🥔 Potato Leaf Disease Classifier")
st.write("Upload a leaf image to classify it via the FastAPI inference backend.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    if st.button("Predict Disease", type="primary", use_container_width=True):
        with st.spinner("Sending image to backend API..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }
                response = requests.post(API_URL, files=files, timeout=30)
                response.raise_for_status()
                data = response.json()

                clean_class = data["class"].replace("Potato___", "").replace("_", " ")
                confidence = data["confidence"]

                st.divider()
                if "healthy" in clean_class.lower():
                    st.success(f"**Prediction:** {clean_class} ({confidence:.2f}% confidence)")
                else:
                    st.warning(f"**Prediction:** {clean_class} ({confidence:.2f}% confidence)")

                st.subheader("Class Probabilities")
                for raw_label, prob_pct in data["probabilities"].items():
                    label = raw_label.replace("Potato___", "").replace("_", " ")
                    st.write(f"**{label}:** {prob_pct:.2f}%")
                    st.progress(prob_pct / 100.0)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend API. Make sure `uvicorn backend:app` is running on port 8000.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")