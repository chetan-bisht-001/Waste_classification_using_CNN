import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import altair as alt
import google.generativeai as genai  # Gemini API

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="AI Waste Classifier", page_icon="♻️", layout="wide")

# -----------------------------
# Configure Gemini API
# -----------------------------
GEMINI_API_KEY = "AIzaSyAAzD-xm96fZ5IOEaxlNdJXoK_4q5jOCsU"  
genai.configure(api_key=GEMINI_API_KEY)

# Load Gemini Model
llm_model = genai.GenerativeModel("gemini-1.5-flash")

# -----------------------------
# Load Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("waste_classifier_mobilenetv2_unfreezelast30 .h5")

try:
    model = load_model()
except Exception as e:
    
    st.error("⚠️ Could not load model. Please check the model file path.")
    st.stop()

# -----------------------------
# Define Classes
# -----------------------------
class_labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# -----------------------------
# Title & Description
# -----------------------------
st.title("♻️ AI Waste Classifier")
st.markdown(
    """
    Upload an image of waste (plastic, paper, glass, etc.), and the AI will classify it.  
    This tool can help with **smart recycling** and **waste management**.
    """
)
st.caption("💡 Tip: You can drag and drop your image here.")

# -----------------------------
# Image Preprocessing
# -----------------------------
@st.cache_data
def preprocess_image(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img

# -----------------------------
# Prediction Function
# -----------------------------
def predict_image(image):
    img_array, img = preprocess_image(image)
    prediction = model.predict(img_array)
    return prediction, img

# -----------------------------
# Recycling Tips with Gemini
# -----------------------------
def generate_recycle_tips(category):
    prompt = f"Give some short and practical recycling tips for {category} waste. Keep them simple and useful for daily life. and and tell how we can use this or maybe what we can make from this "
    try:
        response = llm_model.generate_content(prompt)
        tips_text = response.text.strip()
        tips = [tip.strip("-• ") for tip in tips_text.split("\n") if tip.strip()]
        return tips[:] if tips else ["No tips available."]
    except Exception as e:
        return ["⚠️ Error fetching tips from Gemini API."]

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload an image of waste", type=["jpg", "jpeg", "png"])

# -----------------------------
# Run Prediction
# -----------------------------
if uploaded_file is not None:
    prediction, img = predict_image(uploaded_file)
    probs = prediction[0]
    predicted_label = class_labels[np.argmax(probs)]
    confidence = np.max(probs)

    # Layout: Image on left, Results on right
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.subheader("🔮 Prediction")

        if confidence > 0.8:
            conf_text = "🟢 High Confidence"
        elif confidence > 0.5:
            conf_text = "🟡 Medium Confidence"
        else:
            conf_text = "🔴 Low Confidence"

        st.success(f"✅ Predicted Category: **{predicted_label.upper()}**")
        st.write(f"**Confidence:** {confidence*100:.1f}% ({conf_text})")

        # Probability chart
        st.subheader("📊 Category Probabilities")
        df = pd.DataFrame({"Category": class_labels, "Confidence": probs})
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("Category", sort=None),
            y="Confidence",
            color="Category"
        ).properties(height=300).interactive()

        text = chart.mark_text(
            align="center",
            baseline="bottom",
            dy=-5
        ).encode(text=alt.Text("Confidence", format=".2f"))

        st.altair_chart(chart + text, use_container_width=True)

    # -----------------------------
    # Dynamic Recycling Tips
    # -----------------------------
    st.subheader("🌍 Recycling Tips")
    tips = generate_recycle_tips(predicted_label)
    for tip in tips:
        st.write("✅ " + tip)
