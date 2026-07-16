import os

import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError


API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000/predict/"
)


st.set_page_config(
    page_title="Structural Defect Detection AI",
    page_icon="🏗️",
    layout="centered"
)

st.title("🏗️ Structural Defect Detection AI")

st.write(
    "Upload an image of concrete or masonry to check whether "
    "the AI model detects a visible crack."
)


uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption=uploaded_file.name,
            use_container_width=True
        )

    except (UnidentifiedImageError, OSError):
        st.error(
            "❌ The selected file is not a valid readable image."
        )
        st.stop()

    if st.button(
        "🔍 Analyze Structure",
        type="primary",
        use_container_width=True
    ):

        img_bytes = uploaded_file.getvalue()

        files = {
            "file": (
                uploaded_file.name,
                img_bytes,
                uploaded_file.type or "application/octet-stream"
            )
        }

        try:
            with st.spinner("AI is inspecting the image..."):

                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=30
                )

            if response.status_code == 200:

                result = response.json()

                prediction = result.get(
                    "prediction",
                    "Unknown prediction"
                )

                confidence = result.get(
                    "confidence",
                    0
                )

                processing_time = result.get(
                    "processing_time_ms"
                )

                filename = result.get(
                    "filename",
                    uploaded_file.name
                )

                if isinstance(confidence, str):
                    confidence = float(
                        confidence.strip("%")
                    )

                confidence = float(confidence)
                confidence_text = f"{confidence:.2f}%"

                if "Positive" in prediction:

                    st.error(
                        f"🚨 **POTENTIAL CRACK DETECTED**  \n"
                        f"Confidence: **{confidence_text}**"
                    )

                    st.warning(
                        "This is a model finding, not definite proof "
                        "of structural damage. Flag the area for "
                        "manual inspection."
                    )

                else:

                    st.success(
                        f"✅ **NO VISIBLE CRACK DETECTED**  \n"
                        f"Confidence: **{confidence_text}**"
                    )

                    st.info(
                        "No visible crack was detected by the model. "
                        "Safety-critical cases should still be "
                        "inspected manually."
                    )

                confidence_column, time_column = st.columns(2)

                with confidence_column:
                    st.metric(
                        label="Model Confidence",
                        value=confidence_text
                    )

                with time_column:
                    if processing_time is not None:
                        st.metric(
                            label="Processing Time",
                            value=f"{float(processing_time):.2f} ms"
                        )
                    else:
                        st.metric(
                            label="Processing Time",
                            value="Not available"
                        )

                with st.expander("View prediction details"):

                    st.write(f"**Filename:** `{filename}`")
                    st.write(f"**Model output:** `{prediction}`")
                    st.write(
                        f"**Confidence:** `{confidence_text}`"
                    )

                    if processing_time is not None:
                        st.write(
                            f"**Processing time:** "
                            f"`{float(processing_time):.2f} ms`"
                        )

            elif response.status_code == 400:
                st.error(
                    "❌ Invalid or corrupted image."
                )

            elif response.status_code == 413:
                st.error(
                    "❌ The uploaded image is too large."
                )

            elif response.status_code == 415:
                st.error(
                    "❌ Unsupported file type. "
                    "Please upload a JPG or PNG image."
                )

            elif response.status_code == 422:
                st.error(
                    "❌ Missing or invalid upload request."
                )

            elif response.status_code >= 500:
                st.error(
                    "❌ The AI server encountered an unexpected "
                    "problem. Please try again later."
                )

            else:
                try:
                    error_detail = response.json().get(
                        "detail",
                        "Unknown server error."
                    )
                except ValueError:
                    error_detail = "Unknown server error."

                st.error(
                    f"❌ Server error {response.status_code}: "
                    f"{error_detail}"
                )

        except requests.exceptions.Timeout:
            st.error(
                "⏳ The request timed out. The AI server is "
                "taking too long to respond."
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "🚨 Cannot connect to the AI server. "
                "Check whether the FastAPI Docker container "
                "is running on port 8000."
            )

        except requests.exceptions.RequestException as error:
            st.error(
                f"❌ The request failed: {error}"
            )

        except (ValueError, TypeError, KeyError):
            st.error(
                "❌ The AI server returned an unexpected "
                "response format."
            )


st.divider()

with st.expander("ℹ️ About this system"):
    st.write("**Model architecture:** ResNet18")
    st.write("**Task:** Binary crack classification")
    st.write("**Classes:** Crack / No Crack")
    st.write("**Backend:** FastAPI")
    st.write("**Model serving:** Docker")
    st.write("**Frontend:** Streamlit")
