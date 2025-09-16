import streamlit as st
import cv2
import numpy as np
import os
from moviepy import VideoFileClip
from tempfile import NamedTemporaryFile

# ============================
# Page Configuration
# ============================
st.set_page_config(
    page_title="Deepfake Detection Demo",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================
# Video Conversion Helper
# ============================
def convert_to_mp4(input_file):
    """Converts the uploaded video file to MP4 format safely."""
    # Read the file into memory
    video_bytes = input_file.read()

    # Use a temporary file to handle the conversion
    with NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input_file:
        temp_input_file.write(video_bytes)
        input_path = temp_input_file.name

    # Check if it's already an MP4
    if input_path.lower().endswith(".mp4"):
        return input_path

    output_path = os.path.splitext(input_path)[0] + "_converted.mp4"
    try:
        # Perform the conversion
        clip = VideoFileClip(input_path)
        clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            threads=2,
            logger=None # Suppress verbose output
        )
        clip.close()
        # Clean up the original temp file
        os.remove(input_path)
        return output_path
    except Exception as e:
        st.error(f"⚠️ Video conversion failed: {e}")
        # Clean up if conversion fails
        os.remove(input_path)
        return None


# ============================
# Simple OpenCV-based Analyzer (Heuristic Model)
# ============================
def run_inference(video_path, frame_sample_rate=30):
    """
    A simple heuristic-based analyzer using blur detection.
    NOTE: This is a placeholder for a real deep learning model.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    blur_scores = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    progress_bar = st.progress(0, text="Analyzing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample every Nth frame for efficiency
        if frame_count % frame_sample_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_scores.append(laplacian_var)

        frame_count += 1
        # Update progress bar
        progress_bar.progress(frame_count / total_frames, text=f"Analyzing frame {frame_count}/{total_frames}")

    cap.release()
    progress_bar.empty() # Remove the progress bar after completion

    if not blur_scores:
        return {"fake_probability": 0.0, "real_probability": 1.0}

    # Heuristic: Lower average blur variance might suggest a smoothed-over (fake) video.
    # This is a simple rule and not a reliable deepfake detector.
    avg_blur = np.mean(blur_scores)
    fake_prob = 1 / (1 + np.exp((avg_blur - 100) / 20)) # Logistic scaling around a blur threshold of 100
    real_prob = 1 - fake_prob

    return {
        "fake_probability": float(fake_prob),
        "real_probability": float(real_prob),
    }


# ============================
# Streamlit UI
# ============================
def main():
    # --- Sidebar ---
    with st.sidebar:
        st.title("🎭 Deepfake Detector")
        st.markdown("---")
        st.markdown(
            "**About This App**\n\n"
            "This application provides a demonstration of deepfake detection. "
            "Upload a video, and the system will analyze it to determine the likelihood of it being a deepfake."
        )
        st.markdown(
            "**How It Works**\n\n"
            "The current analysis is based on a simple heuristic model that measures video sharpness (Laplacian variance). "
            "Smoother, less detailed videos are flagged as potentially fake. *This is a placeholder for a real deep learning model.*"
        )
        st.markdown("---")
        st.info("Created by Mihir Ruwala")

    # --- Main Page ---
    st.title("Deepfake Detection Demo")
    st.markdown("Upload a video file to check if it's a potential deepfake. The model will analyze it and provide a probability score.")

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "avi", "mkv"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # Create two columns for a side-by-side layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Video")
            # Convert and get the path
            video_path = convert_to_mp4(uploaded_file)
            if video_path:
                st.video(video_path)

        with col2:
            st.subheader("Analysis Results")
            if video_path:
                # Show a spinner during analysis
                with st.spinner('Analyzing video... Please wait. ✨'):
                    results = run_inference(video_path)

                st.success("✅ Analysis Complete!")

                # Display a high-level result with color
                fake_prob = results['fake_probability']
                if fake_prob > 0.7:
                    st.error("Result: Likely Deepfake")
                elif fake_prob > 0.4:
                    st.warning("Result: Potentially Deepfake")
                else:
                    st.success("Result: Likely Real")

                # Display probabilities using st.metric for a professional look
                st.metric(label="Fake Probability", value=f"{fake_prob:.2%}")
                st.metric(label="Real Probability", value=f"{results['real_probability']:.2%}")

                # Optional: Add a visual bar chart
                st.bar_chart(data=results)

                # Clean up the temporary video file
                os.remove(video_path)

if __name__ == "__main__":
    # The macOS fix for ffmpeg path is often needed for local development.
    # You may not need this line if ffmpeg is in your system's PATH.
    # os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
    main()