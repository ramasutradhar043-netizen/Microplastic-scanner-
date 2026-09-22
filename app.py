import streamlit as st
import cv2
import numpy as np

# Page Configuration
st.set_page_config(page_title="SMART GREEN-SHIELD", page_icon="🌱", layout="centered")

st.title("🌱 SMART GREEN-SHIELD")
st.subheader("Soil Microplastic Contamination Scanner")
st.write("Upload a clear photo of your dried filter paper/cloth to estimate soil microplastic contamination.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    orig_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Image Processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thresholding to detect particles
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Calculate Area Percentage
    total_pixels = thresh.shape[0] * thresh.shape[1]
    plastic_pixels = cv2.countNonZero(thresh)
    contamination_pct = (plastic_pixels / total_pixels) * 100
    
    # Display Results
    st.subheader("📊 Analysis Results")
    st.image(orig_image, caption="Uploaded Filter Paper Sample", use_container_width=True)
    
    st.metric(label="Estimated Microplastic Area Coverage", value=f"{contamination_pct:.2f} %")
    
    # Risk Assessment Logic
    if contamination_pct < 0.5:
        st.success("🟢 LOW RISK: Soil contamination is very low. Safe for regular farming.")
    elif 0.5 <= contamination_pct <= 2.5:
        st.warning("🟡 MODERATE RISK: Moderate microplastic presence detected. Organic Bio-Mulching is recommended to protect crops.")
    else:
        st.error("🔴 HIGH RISK: Severe microplastic contamination detected! Soil remediation and SMART GREEN-SHIELD bio-mulch protection strongly advised.")
      
