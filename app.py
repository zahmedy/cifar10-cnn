"""
Streamlit app for CIFAR-10 inference.

Run locally:
    pip install streamlit pillow torch torchvision
    streamlit run app.py
Upload an image and see the predicted CIFAR-10 class and probabilities.
"""

from pathlib import Path
from typing import Dict

import streamlit as st
from PIL import Image

from config import CHECKPOINT_DIR, CLASSES
from inference import _load_checkpoint, predict_image


@st.cache_resource
def load_model():
    # Load the trained model once and reuse it between interactions.
    checkpoint_path = CHECKPOINT_DIR / "cifar10_cnn.pt"
    return _load_checkpoint(checkpoint_path)


def run_inference(image: Image.Image) -> Dict[str, float]:
    # Save the uploaded image to a temporary file and classify it.
    temp_path = Path("temp_upload.png")
    image.save(temp_path)
    model = load_model()
    label, probs = predict_image(model, temp_path)
    return {"label": label, "probs": {cls: float(probs[i]) for i, cls in enumerate(CLASSES)}}


def main(): 
    st.set_page_config(page_title="CIFAR-10 CNN", page_icon="🖼️")
    st.title("CIFAR-10 CNN Demo")
    st.write("Upload an image to see the predicted CIFAR-10 class and probability breakdown.")

    uploaded_file = st.file_uploader("Upload an image (will be resized to 32x32)", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)

        try:
            with st.spinner("Running inference..."):
                result = run_inference(image)
            st.success(f"Prediction: {result['label']}")
            st.bar_chart(result["probs"])
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong: {e}")


if __name__ == "__main__":
    main()
