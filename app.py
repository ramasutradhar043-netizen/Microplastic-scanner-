import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io

st.set_page_config(
    page_title="SMART GREEN-SHIELD - Microplastic Scanner", page_icon="🌱"
)

st.markdown(
    """
    <div style='background-color: #1e3d2f; padding: 20px; border-radius: 10px; text-align: center;'>
        <h1 style='color: #ffffff; margin: 0;'>🌱 SMART GREEN-SHIELD</h1>
        <p style='color: #a3e4d7; font-size: 18px; margin-top: 5px;'>Soil Microplastic Contamination Scanner</p>
    </div>
""",
    unsafe_allow_html=True,
)

st.write("")
st.markdown(
    "Upload a clear photo of your dried filter paper/cloth against a plain white"
    " background to estimate soil microplastic contamination."
)

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        # --- FIX: Robust Image Loading ---
        # Read file bytes directly to handle potential corruption or format issues
        file_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(file_bytes))

        # Convert PIL Image to OpenCV format (RGB)
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Display the image in the app
        st.image(image, caption="Uploaded Filter Sample", use_column_width=True)

        if st.button("Run Microplastic Analysis"):
            with st.spinner("Processing image and filtering background noise..."):
                # Convert to grayscale for OpenCV processing
                gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

                # Apply Gaussian Blur to remove high-frequency noise and minor glare
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)

                # Adaptive thresholding to handle lighting variations
                thresh = cv2.adaptiveThreshold(
                    blurred,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    11,
                    2,
                )

                # Find contours of particles
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Filter contours by minimum area
                min_particle_area = 30
                valid_particles = 0
                total_particle_pixels = 0

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > min_particle_area:
                        valid_particles += 1
                        total_particle_pixels += area

                total_image_pixels = gray.shape[0] * gray.shape[1]
                
                # Safely handle division by zero if image is somehow empty
                if total_image_pixels == 0:
                     coverage_percentage = 0.0
                else:
                     coverage_percentage = min(
                        100.0, (total_particle_pixels / total_image_pixels) * 100 * 3.5
                    )

                # Determine Risk Level based on filtered coverage
                if coverage_percentage < 5:
                    risk_level = "Low Risk"
                    color = "green"
                elif coverage_percentage < 20:
                    risk_level = "Moderate Risk"
                    color = "orange"
                else:
                    risk_level = "High Risk"
                    color = "red"

                st.markdown("---")
                st.subheader("📊 Analysis Results")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label="Estimated Coverage", value=f"{coverage_percentage:.2f}%"
                    )
                    st.metric(label="Detected Particles", value=str(valid_particles))

                with col2:
                    st.markdown(f"### Risk Level: :{color}[{risk_level}]")

                if risk_level == "High Risk":
                    st.error(
                        "High microplastic contamination detected! Consider implementing"
                        " bio-mulch sheets and remediation steps."
                    )
                elif risk_level == "Moderate Risk":
                    st.warning(
                        "Moderate contamination observed. Monitor soil health regularly."
                    )
                else:
                    st.success(
                        "Low contamination levels observed. Soil sample is relatively"
                        " clean."
                    )

    except Exception as e:
        st.error(f"An error occurred while loading the image: {e}")
        st.info("Please ensure you are uploading a valid JPG or PNG image file.")

