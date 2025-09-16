import streamlit as st
import multiprocessing as mp
import tempfile
import os

# 🔽 Your ML imports here
# import torch
# from facenet_pytorch import MTCNN
# from my_model import load_model, classify_video


def run_inference(video_path):
    """
    Dummy placeholder for your deepfake detection pipeline.
    Replace with your actual MTCNN + CNN + GNN inference logic.
    """
    # Example: run face detection + classification
    # result = classify_video(video_path)
    # return result

    # Temporary placeholder
    return "Real (confidence: 91%)"


def main():
    st.title("Deepfake Detection")
    st.write("A deepfake detection project by Team Aegis using face-based CNN+GNN classification with MTCNN.")

    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_file:
        st.video(uploaded_file)

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name

        # Button to trigger detection
        if st.button("🔍 Run Deepfake Detection"):
            with st.spinner("Analyzing video..."):
                result = run_inference(temp_video_path)

            st.success(f"✅ Detection Result: {result}")

            # Cleanup temp files 
            os.remove(temp_video_path)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()