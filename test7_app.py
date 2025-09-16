# final version of website without gui features
import streamlit as st
import moviepy as mp
import cv2
import numpy as np
import os

# ============================
# Video Conversion Helper
# ============================
from moviepy import VideoFileClip
import os
import streamlit as st

def convert_to_mp4(input_path):
    """MoviePy conversion without verbose/logger (safe across versions)."""
    if input_path.lower().endswith(".mp4"):
        return input_path

    output_path = os.path.splitext(input_path)[0] + "_converted.mp4"
    try:
        clip = VideoFileClip(input_path)
        # <-- no verbose/logger arguments here
        clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            threads=1   # helps avoid macOS multiprocessing/mutex crashes
        )
        clip.close()
        return output_path
    except Exception as e:
        st.error(f"⚠️ Video conversion failed: {e}")
        return None


# ============================
# Simple OpenCV-based Analyzer
# ============================
def run_inference(video_path, frame_sample_rate=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Could not read video with OpenCV.")
        return {"fake_probability": 0.0, "real_probability": 0.0}

    blur_scores = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample every Nth frame
        if frame_count % frame_sample_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # blur score
            blur_scores.append(laplacian_var)

        frame_count += 1

    cap.release()

    if len(blur_scores) == 0:
        return {"fake_probability": 0.0, "real_probability": 0.0}

    avg_blur = np.mean(blur_scores)

    # 🔑 Heuristic: lower blur variance => possibly fake
    fake_prob = 1 / (1 + np.exp((avg_blur - 100) / 20))  # logistic scaling
    real_prob = 1 - fake_prob

    return {
        "fake_probability": float(fake_prob),
        "real_probability": float(real_prob),
    }


# ============================
# Streamlit UI
# ============================
def main():
    st.title("🎭 Deepfake Detection Demo")
    st.markdown("-created by Mihir Ruwala")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_video", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Convert to mp4
        st.info("Converting video to .mp4...")
        video_path = convert_to_mp4("temp_video")

        if video_path:
            st.video(video_path)

            # Run inference
            st.info("Analyzing video frames with OpenCV...")
            results = run_inference(video_path)

            # Display results
            st.success("✅ Analysis complete!")
            st.write(f"**Fake Probability:** {results['fake_probability']:.2f}")
            st.write(f"**Real Probability:** {results['real_probability']:.2f}")


if __name__ == "__main__":
    os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"  # macOS fix
    main()