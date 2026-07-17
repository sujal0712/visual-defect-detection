from pathlib import Path
import time

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


st.set_page_config(
    page_title="Structural Defect Detection",
    page_icon="🏗️",
    layout="centered",
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT
    / "saved_weights"
    / "resnet18_defect_model.pth"
)

CLASS_NAMES = [
    "Negative (No Crack)",
    "Positive (Crack Detected)",
]


image_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


@st.cache_resource
def load_model() -> torch.nn.Module:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights were not found at: {MODEL_PATH}"
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    # Supports checkpoints saved as:
    # {"model_state_dict": ...}
    if (
        isinstance(state_dict, dict)
        and "model_state_dict" in state_dict
    ):
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.eval()

    return model


def predict_image(
    image: Image.Image,
    model: torch.nn.Module,
) -> tuple[str, float, float]:
    started_at = time.perf_counter()

    image = image.convert("RGB")
    tensor = image_transform(image).unsqueeze(0)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_index].item()) * 100

    processing_time_ms = (
        time.perf_counter() - started_at
    ) * 1000

    return (
        CLASS_NAMES[predicted_index],
        confidence,
        processing_time_ms,
    )


st.title("🏗️ Structural Defect Detection")
st.write(
    "Upload an image of a concrete surface to check whether "
    "the model detects a visible crack."
)

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the AI model: {error}")
    st.stop()


uploaded_file = st.file_uploader(
    "Upload a concrete surface image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image.load()
    except (UnidentifiedImageError, OSError):
        st.error("The uploaded file is not a valid image.")
        st.stop()

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True,
    )

    if st.button(
        "Analyze Structure",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Analyzing the image..."):
            prediction, confidence, processing_time = predict_image(
                image,
                model,
            )

        if prediction == "Positive (Crack Detected)":
            st.error(f"⚠️ {prediction}")
        else:
            st.success(f"✅ {prediction}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Confidence",
                f"{confidence:.2f}%",
            )

        with col2:
            st.metric(
                "Processing time",
                f"{processing_time:.0f} ms",
            )

        st.warning(
            "This is an AI prototype and should not replace "
            "a professional structural inspection."
        )