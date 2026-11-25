import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

from weed_info import (
    WEEDS,
    GENERIC_WEED,
    GENERIC_CROP,
    HERBICIDE_SAFETY,
    DISCLAIMER
)

st.set_page_config(
    page_title="AI Crop and Weed Detection",
    page_icon="🌱",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
CONFIDENCE_THRESHOLD = 0.60

CROPS = ["Common wheat", "Maize", "Sugar beet"]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        BASE_DIR / "species_model.keras"
    )


model = load_model()

with open(BASE_DIR / "species_class_names.json", "r") as f:
    class_names = json.load(f)


def show_list(title, items):
    st.markdown(f"### {title}")
    for item in items:
        st.markdown(f"- {item}")


def show_weed_details(species):
    if species in WEEDS:
        info = WEEDS[species]

        st.subheader(info["common_name"])
        st.write("Hindi Name:", info["hindi_name"])
        st.write("Scientific Name:", info["scientific_name"])
        st.write("Risk Level:", info["risk"])
        st.write(info["description"])

        st.warning(info["health_hazard"])

        show_list("🛡️ Precautions", info["precautions"])
        show_list("🧰 Control Methods", info["control"])

    else:
        st.info(
            "Species identified. General weed precautions "
            "and control methods are provided below."
        )

        show_list(
            "🛡️ Precautions",
            GENERIC_WEED["precautions"]
        )

        show_list(
            "🧰 Control Methods",
            GENERIC_WEED["control"]
        )


st.title("🌱 AI-Based Crop and Weed Species Detection")

st.write(
    "Upload a plant image to identify its species "
    "using a CNN-based deep learning model."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload Plant Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Plant Image",
        use_container_width=True
    )

    if st.button("🔍 Identify Plant"):

        img = image.resize((64, 64))

        img_array = np.array(
            img,
            dtype=np.float32
        )

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        prediction = model.predict(
            img_array,
            verbose=0
        )[0]

        predicted_index = np.argmax(prediction)

        species = class_names[predicted_index]

        confidence = float(
            prediction[predicted_index]
        )

        st.session_state["result"] = {
            "species": species,
            "confidence": confidence,
            "predictions": prediction
        }

    result = st.session_state.get("result")

    if result:

        species = result["species"]
        confidence = result["confidence"]

        st.divider()

        if confidence < CONFIDENCE_THRESHOLD:

            st.warning(
                f"Uncertain Prediction: {confidence*100:.2f}%. "
                "Please upload a clearer plant image."
            )

        elif species in CROPS:

            st.success(f"🌾 Crop Detected: {species}")

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

            show_list(
                "🛡️ Crop Precautions",
                GENERIC_CROP["precautions"]
            )

        else:

            if species in WEEDS:
                st.error(
                    f"🌿 Weed Detected: "
                    f"{WEEDS[species]['common_name']}"
                )
            else:
                st.error(f"🌿 Weed Detected: {species}")

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

            show_weed_details(species)

            with st.expander("🧪 Herbicide Safety Instructions"):

                for tip in HERBICIDE_SAFETY:
                    st.markdown(f"- {tip}")

        st.caption(DISCLAIMER)

        st.divider()

        st.subheader("Prediction Details")

        for i, score in enumerate(result["predictions"]):
            st.write(
                f"{class_names[i]}: {score*100:.2f}%"
            )

st.divider()

st.caption(
    "Weed Detection using Convolutional Neural Network (CNN)"
)