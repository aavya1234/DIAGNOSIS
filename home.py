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
    {'Diagnosis': 'Brain Tumor Detected',
        'Causes': 'Genetic mutations, radiation exposure, family history, certain chemicals and industrial products, immune system disorders',
        'Prevention': 'Avoiding radiation exposure, protective gear in industrial settings, genetic counseling if family history is known',
        'Diet': 'High fiber foods, fruits, vegetables, lean proteins, avoiding processed foods and sugars',
        'Exercise': 'Moderate aerobic exercise, strength training, flexibility exercises'
    },
    {
        'Diagnosis': 'No Brain Tumor',
        'Causes': 'N/A',
        'Prevention': 'Regular health check-ups, maintaining a healthy lifestyle',
        'Diet': 'Balanced diet rich in fruits, vegetables, whole grains, and lean proteins',
        'Exercise': 'Regular physical activity, a mix of aerobic, strength, and flexibility exercises'
    },
    {
        'Diagnosis': 'Mild Dementia Detected⚕️',
        'Causes': 'Age, family history, genetics, head trauma, lifestyle factors (smoking, alcohol use)',
        'Prevention': 'Healthy diet, regular exercise, cognitive activities, managing cardiovascular risk factors',
        'Diet': 'Mediterranean diet, foods rich in omega-3 fatty acids, antioxidants, and vitamins',
        'Exercise': 'Aerobic exercises, strength training, balance and flexibility exercises'
    },
    {
        'Diagnosis': 'Moderate Dementia Detected⚕️',
        'Causes': 'Age, family history, genetics, head trauma, lifestyle factors (smoking, alcohol use)',
        'Prevention': 'Healthy diet, regular exercise, cognitive activities, managing cardiovascular risk factors',
        'Diet': 'Mediterranean diet, foods rich in omega-3 fatty acids, antioxidants, and vitamins',
        'Exercise': 'Aerobic exercises, strength training, balance and flexibility exercises'
    },
    {
        'Diagnosis': 'No Dementia Detected⚕️',
        'Causes': 'N/A',
        'Prevention': 'Healthy lifestyle, regular cognitive and physical activities',
        'Diet': 'Balanced diet rich in fruits, vegetables, whole grains, and lean proteins',
        'Exercise': 'Regular physical activity, a mix of aerobic, strength, and flexibility exercises'
    },
    {
        'Diagnosis': 'Very Mild Dementia Detected⚕️',
        'Causes': 'Age, family history, genetics, head trauma, lifestyle factors (smoking, alcohol use)',
        'Prevention': 'Healthy diet, regular exercise, cognitive activities, managing cardiovascular risk factors',
        'Diet': 'Mediterranean diet, foods rich in omega-3 fatty acids, antioxidants, and vitamins',
        'Exercise': 'Aerobic exercises, strength training, balance and flexibility exercises'
    },
    {
        'Diagnosis': 'Normal Arthritis Detected🔵',
        'Causes': 'Age, joint injury, obesity, genetics, overuse of the joint',
        'Prevention': 'Maintaining healthy weight, regular exercise, protecting joints from injury',
        'Diet': 'Anti-inflammatory diet, rich in omega-3 fatty acids, fruits, vegetables, whole grains',
        'Exercise': 'Low-impact aerobic exercises, strength training, flexibility exercises'
    },
    {
        'Diagnosis': 'Arthritis is Doubtful🔵',
        'Causes': 'Age, joint injury, obesity, genetics, overuse of the joint',
        'Prevention': 'Maintaining healthy weight, regular exercise, protecting joints from injury',
        'Diet': 'Anti-inflammatory diet, rich in omega-3 fatty acids, fruits, vegetables, whole grains',
        'Exercise': 'Low-impact aerobic exercises, strength training, flexibility exercises'
    },
    {
        'Diagnosis': 'Mild Arthritis Detected🔵',
        'Causes': 'Age, joint injury, obesity, genetics, overuse of the joint',
        'Prevention': 'Maintaining healthy weight, regular exercise, protecting joints from injury',
        'Diet': 'Anti-inflammatory diet, rich in omega-3 fatty acids, fruits, vegetables, whole grains',
        'Exercise': 'Low-impact aerobic exercises, strength training, flexibility exercises'
    },
    {
        'Diagnosis': 'Moderate Arthritis Detected🔵',
        'Causes': 'Age, joint injury, obesity, genetics, overuse of the joint',
        'Prevention': 'Maintaining healthy weight, regular exercise, protecting joints from injury',
        'Diet': 'Anti-inflammatory diet, rich in omega-3 fatty acids, fruits, vegetables, whole grains',
        'Exercise': 'Low-impact aerobic exercises, strength training, flexibility exercises'
    },
    {
        'Diagnosis': 'Severe Arthritis Detected🔵',
        'Causes': 'Age, joint injury, obesity, genetics, overuse of the joint',
        'Prevention': 'Maintaining healthy weight, regular exercise, protecting joints from injury',
        'Diet': 'Anti-inflammatory diet, rich in omega-3 fatty acids, fruits, vegetables, whole grains',
        'Exercise': 'Low-impact aerobic exercises, strength training, flexibility exercises'
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


st.set_page_config(
    page_title="Medical Image Diagnosis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="auto",
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
    </style>
    <style>
    .reportview-container .main .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    .reportview-container .main {
        background-color: rgba(0, 0, 0, 0.5); 
    }
    h1 {
        font-size: 48px; 
        color: #FFD700; 
        font-family: 'Arial', sans-serif; 
        text-shadow: 2px 2px 4px rgba(0, 191, 255, 0.5);
        margin-bottom: 20px;
    }
    h2 {
        font-size: 36px; 
        color: #00BFFF; 
        font-family: 'Arial', sans-serif; 
        margin-bottom: 15px; 
    }
    h3 {
        font-size: 28px;
        color: #87CEEB; 
        font-family: 'Arial', sans-serif; 
        margin-bottom: 10px; 
    .sidebar-title {
        font-size: 40px; 
        color: #FFD700;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 191, 255, 0.5);
        margin-bottom: 10px;
        text-align: center;
    }
    
    .sidebar-upload {
        font-size: 18px;
        color: #FFFFFF;
        font-weight: bold;
        margin-bottom: 20px;
    }
    
    a {
        color: #FFD700;
        text-decoration: none;
    }
    a:hover {
        color: #FFFFFF;
        text-decoration: underline;
    }
    hr {
        border: 1px solid #00BFFF;
        margin-top: 20px;
        margin-bottom: 20px;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main heading and logo
st.markdown('<h1 class="custom-header">👨‍⚕️AI-Driven Medical Diagnosis🏥</h1>', unsafe_allow_html=True)

with st.sidebar:
    # Custom CSS for styling the sidebar elements, including the text and logo
    st.markdown(
        """
        <style>
        
        .sidebar-container {
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            background-color: #1E1E2F;
            border-radius: 10px;
            box-shadow: 0px 0px 15px rgba(0, 191, 255, 0.5);
            font-family: 'Arial', sans-serif;
        }
        .sidebar-title {
            font-size: 40px;
            color: #FFD700;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0, 191, 255, 0.5);
            margin-bottom: 10px;
            text-align: center;
        }
        
        
        </style>
        """,
        unsafe_allow_html=True
    )
    # Container for the entire sidebar with the title
    st.markdown('<div class="sidebar-container"><div class="sidebar-title">MedCare🩺</div>', unsafe_allow_html=True)
    st.markdown(" ", unsafe_allow_html=True)
    # File uploader with a title
    st.markdown('<div class="sidebar-upload"><h3>Upload Medical Image📸</h3></div>', unsafe_allow_html=True)
    st.markdown(
    """
    <div class="sidebar-instructions">
        <b>Upload Instructions:</b>
        <ul>
            <li>Click the "Browse files" button to select a medical image from your computer.📤</li>
            <li>Supported file formats: PNG, JPG, JPEG.📋</li>
            <li>Maximum file size: 10MB.⚖️</li>
            <li>Upload a clear and high-quality image for better analysis.✨</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

    st.markdown(" ", unsafe_allow_html=True)
    # Add additional sidebar features like contact info or quick links
    st.markdown(
        """
        <div class="sidebar-feature">
        <i class="fas fa-cog"></i> <h3>Settings⚙️</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="sidebar-contact">
            <h3>Contact Us ☎️</h3>  <a href="mailto:support@medcare.com">support@medcare.com</a>
            <br>
            <a href="https://www.medcare.com" target="_blank">Visit MedCare Website</a>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(" ", unsafe_allow_html=True)

    
    # Close the container div
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown(
    """
    <style>
    .image-container {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    
    .details-container {
        margin: 0 auto;
        max-width: 800px; /* Adjust the max width as needed */
        padding: 10px;
        font-family: 'Arial', sans-serif;
        color: #FFFFFF;
    }
    .details-container h2 {
        color: darkblue; 
        text-align: center;
        font-family:'Helvetica';
    }

    .details-container h2:active, .details-container h2:focus {
        color: #FFFF00; /* Bright yellow color on click */
        box-shadow: 0 0 10px #FFFF00; /* Add a glow effect */
        transition: color 0.2s, box-shadow 0.2s; /* Smooth transition */
    }
   
  
    .details-container h3 {
        color: #00BFFF; /* Bright blue for subheadings */
    }
    .details-container p {
        text-align: left;
        margin: 10px 0;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if uploaded_file is not None:
    im = Image.open(uploaded_file).convert('RGB')
    im_transformed = transform(im).unsqueeze(0)

    # Predict the class using the ONNX model
    ort_inputs = {"input.1": im_transformed.numpy()}  # Match the input name expected by the ONNX model
    ort_session = ort.InferenceSession("model.onnx")
    ort_outs = ort_session.run(None, ort_inputs)[0]

    # Get the class with the highest probability
    class_no = np.argmax(ort_outs)
   
    # Display the uploaded image centered
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image(im, caption="Uploaded Image", use_column_width=False, width=300)
    st.markdown('</div>', unsafe_allow_html=True)
    
 
    # Display the diagnosis and detailed information
    diagnosis = get_class_name(class_no)
    detailed_info = get_detailed_info(class_no)

    st.markdown(f'<div class="details-container">', unsafe_allow_html=True)
    st.markdown(f"<h2>{diagnosis}</h2>", unsafe_allow_html=True)
    
    st.markdown(f"<h3>🔬Causes:</h3><p>{detailed_info['Causes']}</p>", unsafe_allow_html=True)
    st.markdown(f"<h3>🛡️Prevention:</h3><p>{detailed_info['Prevention']}</p>", unsafe_allow_html=True)
    st.markdown(f"<h3>🥗Diet:</h3><p>{detailed_info['Diet']}</p>", unsafe_allow_html=True)
    st.markdown(f"<h3>🏃‍♂️Exercise:</h3><p>{detailed_info['Exercise']}</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .image-container {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .image-container img {
        box-shadow: 0 0 10px rgba(255, 255, 0, 0.5); /* Add a yellow glow effect */
        transition: box-shadow 0.2s; /* Smooth transition */
    }

    .image-container img:hover {
        box-shadow: 0 0 20px rgba(255, 255, 0, 0.8); /* Increase glow effect on hover */
    }
    .details-container {
        margin: 0 auto;
        max-width: 800px; /* Adjust the max width as needed */
        padding: 10px;
        font-family: 'Arial', sans-serif;
        color: #FFFFFF;
    }
    .details-container h2 {
        color: darkblue; /* Initial color */
        text-align: center;
    }

    .details-container h2:active, .details-container h2:focus {
        color: #FFFF00; /* Bright yellow color on click */
        box-shadow: 0 0 10px #FFFF00; /* Add a glow effect */
        transition: color 0.2s, box-shadow 0.2s; /* Smooth transition */
    }
   
    
    .details-container h3 {
        color: #00BFFF; /* Bright blue for subheadings */
    }
    .details-container p {
        text-align: left;
        margin: 10px 0;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)