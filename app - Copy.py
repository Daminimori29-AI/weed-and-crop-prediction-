import json
import os

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

from weed_info import (
    WEEDS,
    GENERIC_WEED,
    GENERIC_CROP,
    HERBICIDE_SAFETY,
    DISCLAIMER,
)

# Page Configuration
st.set_page_config(
    page_title="Weed Detection System",
    page_icon="🌱",
    layout="wide"
)

CONFIDENCE_THRESHOLD = 0.6


# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")


# Class names: for the current binary model this is ["crop", "weed"].
# After training a multi-class model, put the species names in
# class_names.json in the SAME order as the training labels.
@st.cache_data
def load_class_names(n_outputs):
    if os.path.exists("class_names.json"):
        with open("class_names.json", encoding="utf-8") as f:
            names = json.load(f)
        if len(names) == n_outputs:
            return names
    return ["crop", "weed"] if n_outputs == 2 else [f"class_{i}" for i in range(n_outputs)]


def show_list(title, items):
    st.markdown(f"**{title}**")
    for item in items:
        st.markdown(f"- {item}")


def show_weed_details(key):
    info = WEEDS[key]
    st.markdown(f"### 🌿 {info['common_name']}")
    st.write(f"**Hindi name:** {info['hindi_name']}")
    st.write(f"**Scientific name:** *{info['scientific_name']}*")
    st.write(f"**Risk level:** {info['risk']}")
    st.write(info["description"])
    st.warning(f"⚠️ Health / livestock hazard: {info['health_hazard']}")
    show_list("🛡️ Precautions", info["precautions"])
    show_list("🧰 Control methods", info["control"])


model = load_model()

# Heading
st.title("🌱 Weed Detection System")
st.write("AI-Based Crop and Weed Classification using CNN")

st.divider()

# Image Upload
uploaded_file = st.file_uploader(
    "Upload a Plant Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Unable to read image.")

    else:

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        with col2:

            st.subheader("Detection")

            if st.button("🔍 Detect Plant"):

                resized = cv2.resize(img, (64, 64))
                normalized = resized / 255.0
                input_image = np.expand_dims(normalized, axis=0)

                prediction = model.predict(input_image, verbose=0)[0]
                class_names = load_class_names(len(prediction))

                top_idx = int(np.argmax(prediction))
                confidence = float(prediction[top_idx])
                top_name = class_names[top_idx]

                st.divider()

                # Store result so the weed-species picker below survives reruns
                st.session_state["result"] = {
                    "name": top_name,
                    "confidence": confidence,
                    "probs": {class_names[i]: float(p) for i, p in enumerate(prediction)},
                }

            result = st.session_state.get("result")

            if result:
                name = result["name"]
                confidence = result["confidence"]

                if confidence < CONFIDENCE_THRESHOLD:
                    st.warning(
                        f"Uncertain Prediction: {confidence*100:.2f}%. "
                        "Please upload a clearer, closer photo of a single plant."
                    )

                elif name == "crop":
                    st.success("🌾 Crop Detected!")
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                    show_list("🛡️ Precautions", GENERIC_CROP["precautions"])

                elif name == "weed":
                    # Binary model: species unknown -> generic advice
                    st.error("🌿 Weed Detected!")
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                    st.info(GENERIC_WEED["note"])
                    show_list("🛡️ Precautions", GENERIC_WEED["precautions"])
                    show_list("🧰 Control methods", GENERIC_WEED["control"])

                    st.divider()
                    chosen = st.selectbox(
                        "Know the weed species? Select it for specific tips:",
                        ["-- Select --"] + list(WEEDS.keys()),
                    )
                    if chosen != "-- Select --":
                        show_weed_details(chosen)

                elif name in WEEDS:
                    # Multi-class model: species identified by the model
                    st.error(f"🌿 Weed Detected: {WEEDS[name]['common_name']}")
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                    show_weed_details(name)

                else:
                    st.error(f"🌿 Detected: {name}")
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                    st.info("No details stored for this class yet. Add it to weed_info.py.")

                if name != "crop" and confidence >= CONFIDENCE_THRESHOLD:
                    with st.expander("🧪 Herbicide safety (if you spray)"):
                        for tip in HERBICIDE_SAFETY:
                            st.markdown(f"- {tip}")

                st.caption(DISCLAIMER)

                st.write("### Prediction Details")
                for cls, p in result["probs"].items():
                    st.write(f"{cls.capitalize()}: {p*100:.2f}%")

st.divider()

st.caption(
    "Weed Detection using Convolutional Neural Network (CNN)"
)
