import os
import io
from datetime import datetime

import numpy as np
import onnx
import onnxruntime as ort
import streamlit as st
import torch
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage,
)
from reportlab.lib.styles import getSampleStyleSheet
from torchvision import transforms

# =========================
#  CLASS NAMES & DETAILS
# =========================

class_names = [
    "Brain Tumor Detected",
    "No Brain Tumor",
    "Mild Dementia Detected",
    "Moderate Dementia Detected",
    "No Dementia Detected",
    "Very Mild Dementia Detected",
    "Normal Arthritis",
    "Doubtful Arthritis",
    "Mild Arthritis",
    "Moderate Arthritis",
    "Severe Arthritis",
]

detailed_info = [
    {
        "Diagnosis": "Brain Tumor Detected",
        "Causes": "Genetic mutations, radiation exposure, family history, certain chemicals and industrial products, immune system disorders",
        "Prevention": "Avoiding radiation exposure, protective gear in industrial settings, genetic counseling if family history is known",
        "Diet": "High fiber foods, fruits, vegetables, lean proteins, avoiding processed foods and sugars",
        "Exercise": "Moderate aerobic exercise, strength training, flexibility exercises",
    },
    {
        "Diagnosis": "No Brain Tumor",
        "Causes": "N/A",
        "Prevention": "Regular health check-ups, maintaining a healthy lifestyle",
        "Diet": "Balanced diet rich in fruits, vegetables, whole grains, and lean proteins",
        "Exercise": "Regular physical activity, a mix of aerobic, strength, and flexibility exercises",
    },
    {
        "Diagnosis": "Mild Dementia Detected⚕️",
        "Causes": "Age, family history, genetics, head trauma, lifestyle factors (smoking, alcohol use)",
        "Prevention": "Healthy diet, regular exercise, cognitive activities, managing cardiovascular risk factors",
        "Diet": "Mediterranean diet, foods rich in omega-3 fatty acids, antioxidants, and vitamins",
        "Exercise": "Aerobic exercises, strength training, balance and flexibility exercises",
    },
    {
        "Diagnosis": "Moderate Dementia Detected⚕️",
        "Causes": "Age, family history, genetics, head trauma, lifestyle factors (smoking, alcohol use)",
        "Prevention": "Healthy diet, regular exercise, cognitive activities, managing cardiovascular risk factors",
        "Diet": "Mediterranean diet, foods rich in omega-3 fatty acids, antioxidants, and vitamins",
        "Exercise": "Aerobic exercises, strength training, balance and flexibility exercises",
    },
    {
        "Diagnosis": "No Dementia Detected⚕️",
        "Causes": "N/A",
        "Prevention": "Healthy lifestyle, regular cognitive and physical activities",
        "Diet": "Balanced diet rich in fruits, vegetables, whole grains, and lean proteins",
        "Exercise": "Regular physical activity, a mix of aerobic, strength, and flexibility exercises",
    },
    {
        "Diagnosis": "Very Mild Dementia Detected⚕️",
        "Causes": "Age, family history, genetics, head trauma, lifestyle factors (smoking, alcohol use)",
        "Prevention": "Healthy diet, regular exercise, cognitive activities, managing cardiovascular risk factors",
        "Diet": "Mediterranean diet, foods rich in omega-3 fatty acids, antioxidants, and vitamins",
        "Exercise": "Aerobic exercises, strength training, balance and flexibility exercises",
    },
    {
        "Diagnosis": "Normal Arthritis Detected🔵",
        "Causes": "Age, joint injury, obesity, genetics, overuse of the joint",
        "Prevention": "Maintaining healthy weight, regular exercise, protecting joints from injury",
        "Diet": "Anti-inflammatory diet, rich in omega-3 fatty acids, fruits, vegetables, whole grains",
        "Exercise": "Low-impact aerobic exercises, strength training, flexibility exercises",
    },
    {
        "Diagnosis": "Arthritis is Doubtful🔵",
        "Causes": "Age, joint injury, obesity, genetics, overuse of the joint",
        "Prevention": "Maintaining healthy weight, regular exercise, protecting joints from injury",
        "Diet": "Anti-inflammatory diet, rich in omega-3 fatty acids, fruits, vegetables, whole grains",
        "Exercise": "Low-impact aerobic exercises, strength training, flexibility exercises",
    },
    {
        "Diagnosis": "Mild Arthritis Detected🔵",
        "Causes": "Age, joint injury, obesity, genetics, overuse of the joint",
        "Prevention": "Maintaining healthy weight, regular exercise, protecting joints from injury",
        "Diet": "Anti-inflammatory diet, rich in omega-3 fatty acids, fruits, vegetables, whole grains",
        "Exercise": "Low-impact aerobic exercises, strength training, flexibility exercises",
    },
    {
        "Diagnosis": "Moderate Arthritis Detected🔵",
        "Causes": "Age, joint injury, obesity, genetics, overuse of the joint",
        "Prevention": "Maintaining healthy weight, regular exercise, protecting joints from injury",
        "Diet": "Anti-inflammatory diet, rich in omega-3 fatty acids, fruits, vegetables, whole grains",
        "Exercise": "Low-impact aerobic exercises, strength training, flexibility exercises",
    },
    {
        "Diagnosis": "Severe Arthritis Detected🔵",
        "Causes": "Age, joint injury, obesity, genetics, overuse of the joint",
        "Prevention": "Maintaining healthy weight, regular exercise, protecting joints from injury",
        "Diet": "Anti-inflammatory diet, rich in omega-3 fatty acids, fruits, vegetables, whole grains",
        "Exercise": "Low-impact aerobic exercises, strength training, flexibility exercises",
    },
]

# =========================
#  PREPROCESSING
# =========================

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# Temperature scaling value to reduce overconfidence.[web:24]
TEMPERATURE = 2.0


def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def calibrate_confidence(logits, temperature=TEMPERATURE):
    """
    logits: ONNX output; handles [C] or [1, C].
    Returns: per-class probabilities, clipped top prob, index.
    """
    logits = np.asarray(logits)

    if logits.ndim == 1:
        logits = logits[None, :]  # [1, C]

    scaled = logits / float(temperature)
    probs = softmax_np(scaled, axis=1)  # [1, C]
    top_idx = int(np.argmax(probs[0]))
    top_prob = float(np.max(probs[0]))

    # Clip so the UI never shows 100 %.[web:24]
    top_prob_display = min(top_prob, 0.97)

    return probs[0], top_prob_display, top_idx


def get_risk_level(class_no, conf):
    """
    conf in [0, 1]; returns a human readable risk string.
    """
    base_risk = {
        0: "HIGH RISK",
        1: "NO RISK",
        2: "LOW RISK",
        3: "MEDIUM RISK",
        4: "NO RISK",
        5: "VERY LOW RISK",
        6: "NO RISK",
        7: "LOW RISK",
        8: "LOW RISK",
        9: "MEDIUM RISK",
        10: "HIGH RISK",
    }.get(class_no, "UNKNOWN RISK")

    if conf >= 0.85:
        conf_band = "HIGH confidence"
    elif conf >= 0.65:
        conf_band = "MEDIUM confidence"
    else:
        conf_band = "LOW confidence"

    return f"{base_risk} ({conf_band})"


def get_class_name(class_no):
    if 0 <= class_no < len(class_names):
        return class_names[class_no]
    return "Unknown Diagnosis"


def get_detailed_info(class_no):
    if 0 <= class_no < len(detailed_info):
        return detailed_info[class_no]
    return detailed_info[1]


# =========================
#  PDF GENERATION
# =========================

def generate_pdf(image, patient, diagnosis, info, confidence):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>AI Medical Diagnosis Report</b>", styles["Title"]))
    story.append(Paragraph("Generated by <b>MedCare 🩺</b>", styles["Italic"]))
    story.append(Spacer(1, 12))

    table = Table(
        [
            ["Patient Name", patient["name"]],
            ["Age", str(patient["age"])],
            ["Gender", patient["gender"]],
            ["Diagnosis", diagnosis],
            ["Model Confidence", f"{confidence * 100:.2f}%"],
        ],
        colWidths=[150, 300],
    )

    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 12))

    img_buf = io.BytesIO()
    image.save(img_buf, format="PNG")
    img_buf.seek(0)
    story.append(PDFImage(img_buf, width=200, height=200))
    story.append(Spacer(1, 12))

    for key in ["Causes", "Prevention", "Diet", "Exercise"]:
        story.append(Paragraph(f"<b>{key}</b>", styles["Heading3"]))
        story.append(Paragraph(info[key], styles["Normal"]))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "<b>Medical Disclaimer:</b><br/>"
            "This AI-generated report is intended for informational purposes only. "
            "It does not replace professional medical advice, diagnosis, or treatment.",
            styles["Normal"],
        )
    )

    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            f"Report Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Italic"],
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer


# =========================
#  MODEL LOADING
# =========================

@st.cache_resource
def load_model():
    try:
        model = onnx.load("model.onnx")
        inputs_all = [n.name for n in model.graph.input]
        inputs_init = [n.name for n in model.graph.initializer]
        true_inputs = list(set(inputs_all) - set(inputs_init))
        if not true_inputs:
            raise RuntimeError("Could not infer model input name.")
        input_name = true_inputs[0]

        session = ort.InferenceSession("model.onnx")
        return session, input_name
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


# =========================
#  STREAMLIT UI
# =========================

st.set_page_config(
    page_title="Medical Image Diagnosis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    body {
        background-image: url("image.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .reportview-container .main {
        background-color: rgba(0, 0, 0, 0.65);
    }
    h1 {
        font-size: 48px;
        color: #FFD700;
        text-shadow: 2px 2px 4px rgba(0, 191, 255, 0.5);
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar-title {
        font-size: 32px;
        color: #FFD700;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 191, 255, 0.5);
        margin-bottom: 10px;
        text-align: center;
    }
    .image-container img {
        box-shadow: 0 0 20px rgba(255, 255, 0, 0.5);
        border-radius: 12px;
    }
    .metric-card {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        padding: 24px;
        border-radius: 18px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        color: white;
        text-align: center;
        margin-bottom: 16px;
    }
    .stProgress > div > div > div > div {
        background-color: #FFD700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1>👨‍⚕️ AI‑Driven Medical Diagnosis 🏥</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown(
        "<div class='sidebar-title'>MedCare 🩺</div>",
        unsafe_allow_html=True,
    )
    st.header("🧑 Patient Details")
    name = st.text_input("Full Name", placeholder="Enter patient name")
    age = st.number_input("Age", 0, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    st.markdown("### 📸 Upload Medical Image")
    uploaded_file = st.file_uploader(
        "Choose a medical image (PNG/JPG/JPEG)",
        type=["png", "jpg", "jpeg"],
    )

session, input_name = load_model()
if session is None or input_name is None:
    st.warning("Model not loaded. Ensure 'model.onnx' is present.")
    st.stop()

st.sidebar.success("✅ Model ready")

# =========================
#  PREDICTION & LAYOUT
# =========================

if uploaded_file is not None:
    try:
        im = Image.open(uploaded_file).convert("RGB")
        im_tensor = transform(im).unsqueeze(0).numpy().astype(np.float32)

        raw_out = session.run(None, {input_name: im_tensor})[0]
        probs, top_conf, class_no = calibrate_confidence(raw_out, TEMPERATURE)
        diagnosis = get_class_name(class_no)
        info = get_detailed_info(class_no)
        risk_text = get_risk_level(class_no, top_conf)

        with st.container():
            col_img, col_diag = st.columns([1, 2])

            with col_img:
                st.markdown("### 📷 Uploaded Medical Image")
                st.image(im, use_column_width=True)

                st.markdown("### 📊 Confidence Analysis")
                st.progress(float(top_conf))
                st.write(f"Model confidence: **{top_conf * 100:.1f}%**")

            with col_diag:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h2 style="margin-bottom: 8px;">{diagnosis}</h2>
                        <div style="font-size: 20px; color: #FFD700; margin-top: 4px;">
                            🔔 {risk_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("### 🔬 Causes")
                st.write(info["Causes"])

                st.markdown("### 🛡️ Prevention")
                st.write(info["Prevention"])

            with col_right:
                st.markdown("### 🥗 Recommended Diet")
                st.write(info["Diet"])

                st.markdown("### 🏃‍♂️ Exercise Plan")
                st.write(info["Exercise"])

        st.markdown("---")

        if name:
            patient = {"name": name, "age": age, "gender": gender}
            pdf = generate_pdf(
                im,
                patient,
                f"{diagnosis} - {risk_text}",
                info,
                top_conf,
            )
            st.download_button(
                "📄 Download Medical Report (PDF)",
                pdf,
                file_name=f"{name.replace(' ', '_')}_MedCare_Report.pdf",
                mime="application/pdf",
            )
        else:
            st.warning("Enter patient name to enable PDF download.")

    except Exception as e:
        st.error(f"Prediction error: {e}")
else:
    st.info("Upload a medical image to get diagnosis, risk indicator, and confidence score.")
