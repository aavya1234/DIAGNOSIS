import os
import torch
import onnx
import onnxruntime as ort
from torchvision import transforms
from PIL import Image
import streamlit as st
import numpy as np

# Define class names and detailed information for diagnosis
class_names = [
    'Brain Tumor Detected', 'No Brain Tumor', 'Mild Dementia Detected',
    'Moderate Dementia Detected', 'No Dementia Detected', 'Very Mild Dementia Detected',
    'Normal Arthritis', 'Doubtful Arthritis', 'Mild Arthritis',
    'Moderate Arthritis', 'Severe Arthritis'
]

detailed_info = [
    {'diagnosis': 'Brain Tumor Detected',
        'causes': 'Genetic mutations, radiation exposure, family history, certain chemicals and industrial products, immune system disorders',
        'prevention': 'Avoiding radiation exposure, protective gear in industrial settings, genetic counseling if family history is known',
        'diet': 'High fiber foods, fruits, vegetables, lean proteins, avoiding processed foods and sugars',
        'exercise': 'Moderate aerobic exercise, strength training, flexibility exercises'
    },
    {
        'diagnosis': 'No Brain Tumor',
        'causes': 'N/A',
        'prevention': 'Regular health check-ups, maintaining a healthy lifestyle',
        'diet': 'Balanced diet rich in fruits, vegetables, whole grains, and lean proteins',
        'exercise': 'Regular physical activity, a mix of aerobic, strength, and flexibility exercises'
    },
    {
        'diagnosis': 'Mild Dementia Detected',
        'causes': 'Age, family history, genetics, head trauma, lifestyle factors (smoking, alcohol use)',
        'prevention': 'Healthy diet, regular exercise, cognitive activities, managing cardiovascular risk factors',
        'diet': 'Mediterranean diet, foods rich in omega-3 fatty acids, antioxidants, and vitamins',
        'exercise': 'Aerobic exercises, strength training, balance and flexibility exercises'
    },
    {
        'diagnosis': 'Moderate Dementia Detected',
        'causes': 'Age, family history, genetics, head trauma, lifestyle factors (smoking, alcohol use)',
        'prevention': 'Healthy diet, regular exercise, cognitive activities, managing cardiovascular risk factors',
        'diet': 'Mediterranean diet, foods rich in omega-3 fatty acids, antioxidants, and vitamins',
        'exercise': 'Aerobic exercises, strength training, balance and flexibility exercises'
    },
    {
        'diagnosis': 'No Dementia Detected',
        'causes': 'N/A',
        'prevention': 'Healthy lifestyle, regular cognitive and physical activities',
        'diet': 'Balanced diet rich in fruits, vegetables, whole grains, and lean proteins',
        'exercise': 'Regular physical activity, a mix of aerobic, strength, and flexibility exercises'
    },
    {
        'diagnosis': 'Very Mild Dementia Detected',
        'causes': 'Age, family history, genetics, head trauma, lifestyle factors (smoking, alcohol use)',
        'prevention': 'Healthy diet, regular exercise, cognitive activities, managing cardiovascular risk factors',
        'diet': 'Mediterranean diet, foods rich in omega-3 fatty acids, antioxidants, and vitamins',
        'exercise': 'Aerobic exercises, strength training, balance and flexibility exercises'
    },
    {
        'diagnosis': 'Normal Arthritis Detected',
        'causes': 'Age, joint injury, obesity, genetics, overuse of the joint',
        'prevention': 'Maintaining healthy weight, regular exercise, protecting joints from injury',
        'diet': 'Anti-inflammatory diet, rich in omega-3 fatty acids, fruits, vegetables, whole grains',
        'exercise': 'Low-impact aerobic exercises, strength training, flexibility exercises'
    },
    {
        'diagnosis': 'Arthritis is Doubtful',
        'causes': 'Age, joint injury, obesity, genetics, overuse of the joint',
        'prevention': 'Maintaining healthy weight, regular exercise, protecting joints from injury',
        'diet': 'Anti-inflammatory diet, rich in omega-3 fatty acids, fruits, vegetables, whole grains',
        'exercise': 'Low-impact aerobic exercises, strength training, flexibility exercises'
    },
    {
        'diagnosis': 'Mild Arthritis Detected',
        'causes': 'Age, joint injury, obesity, genetics, overuse of the joint',
        'prevention': 'Maintaining healthy weight, regular exercise, protecting joints from injury',
        'diet': 'Anti-inflammatory diet, rich in omega-3 fatty acids, fruits, vegetables, whole grains',
        'exercise': 'Low-impact aerobic exercises, strength training, flexibility exercises'
    },
    {
        'diagnosis': 'Moderate Arthritis Detected',
        'causes': 'Age, joint injury, obesity, genetics, overuse of the joint',
        'prevention': 'Maintaining healthy weight, regular exercise, protecting joints from injury',
        'diet': 'Anti-inflammatory diet, rich in omega-3 fatty acids, fruits, vegetables, whole grains',
        'exercise': 'Low-impact aerobic exercises, strength training, flexibility exercises'
    },
    {
        'diagnosis': 'Severe Arthritis Detected',
        'causes': 'Age, joint injury, obesity, genetics, overuse of the joint',
        'prevention': 'Maintaining healthy weight, regular exercise, protecting joints from injury',
        'diet': 'Anti-inflammatory diet, rich in omega-3 fatty acids, fruits, vegetables, whole grains',
        'exercise': 'Low-impact aerobic exercises, strength training, flexibility exercises'
    }
]

# Load the ONNX model
onnx_model = onnx.load("model.onnx")

# Define the transform for inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to get class name and details
def get_class_name(class_no):
    return class_names[class_no]

def get_detailed_info(class_no):
    return detailed_info[class_no]

# Streamlit UI
st.title("Medical Image Diagnosis")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    im = Image.open(uploaded_file).convert('RGB')
    im_transformed = transform(im).unsqueeze(0)

    # Predict the class using the ONNX model
    ort_inputs = {"input.1": im_transformed.numpy()}  # Match the input name expected by the ONNX model
    ort_session = ort.InferenceSession("model.onnx")
    ort_outs = ort_session.run(None, ort_inputs)[0]

    # Get the class with the highest probability
    class_no = np.argmax(ort_outs)

    # Display the uploaded image
    st.image(im, caption="Uploaded Image", use_column_width=True)

    # Display the diagnosis and detailed information
    diagnosis = get_class_name(class_no)
    detailed_info = get_detailed_info(class_no)

    st.write("Diagnosis: ", diagnosis)
    st.write("Detailed Information:")
    st.write("Causes: ", detailed_info['causes'])
    st.write("Prevention: ", detailed_info['prevention'])
    st.write("Diet: ", detailed_info['diet'])
    st.write("Exercise: ", detailed_info['exercise'])
else:
    st.write("Please upload an image file.")