import cv2
from ultralytics import YOLO
import streamlit as st
import tempfile
import os
import numpy as np
import time
from moviepy import VideoFileClip, ImageSequenceClip

st.image("coforge logo.png", width=100)
st.title('PPE Detection App')

# Container to store timestamps and missing PPE details of violations
ppe_violation_data = []

def check_ppe_compliance(results):
    """Checks if a person detected in a frame has all required PPE.

    Args:
        results (ultralytics.engine.results.Results): YOLOv8 detection results.

    Returns:
        bool: True if compliant (person has all PPE), False otherwise.
        list: List of missing PPE classes.
    """
    ppe_classes = ["helmet", "jacket", "mask"]
    detected_classes = [results.names[int(cls)] for cls in results.boxes.cls]
    if "person" in detected_classes:
        missing_ppe = [ppe for ppe in ppe_classes if ppe not in detected_classes]
        return len(missing_ppe) == 0, missing_ppe
    else:
        return True, []  # No person detected, assume compliant

def process_and_display(model_path, video_path, skip_frames=5):
    """Processes a video with a YOLOv8 model and displays the output in Streamlit.

    Args:
        model_path (str): Path to the YOLOv8 model file (.pt).
        video_path (str): Path to the input video file.
        skip_frames (int, optional): Number of frames to skip between detections.
                                    Defaults to 5 for faster processing.

    Returns:
        str: Path to the processed output video file.
    """
    model = YOLO(model_path)
    video_clip = VideoFileClip(video_path)
    fps = video_clip.fps
    processed_frames = []
    stframe = st.empty()
    progress_bar = st.progress(0)
    total_frames = int(video_clip.fps * video_clip.duration)
    frames_processed = 0

    for i, frame in enumerate(video_clip.iter_frames(fps=fps, dtype="uint8")):
        if i % skip_frames != 0:
            continue

        results = model(frame)

        # Access detection results
        result = results[0] if len(results) > 0 else None

        # Render bounding boxes and check for PPE compliance
        if result is not None:
            annotated_frame = result.plot()

            compliant, missing_ppe = check_ppe_compliance(result)
            if not compliant:
                # warning_message = f"PPE Violation: Missing {', '.join(missing_ppe)}"
                # cv2.putText(annotated_frame, warning_message, (10, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # st.warning(warning_message)  # Display warning in Streamlit

                # Store timestamp and missing PPE details
                timestamp = i / fps
                ppe_violation_data.append({"timestamp": timestamp, "missing_ppe": missing_ppe})
        else:
            annotated_frame = frame

        processed_frames.append(annotated_frame)
        stframe.image(annotated_frame, channels="RGB")

        frames_processed += skip_frames
        progress = frames_processed / total_frames
        progress_bar.progress(progress if progress <= 1 else 1)  # Ensure progress <= 1

    stframe.empty()
    progress_bar.empty()

    output_clip = ImageSequenceClip(processed_frames, fps=fps // skip_frames)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
        output_clip.write_videofile(temp_output.name, codec="libx264", audio=False)
        output_path = temp_output.name

    return output_path

model_path = 'best.pt' # Update with your model path 
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_video.read())
        input_video_path = temp_input.name

    st.write("Processing video... This might take a while.")
    output_video_path = process_and_display(model_path, input_video_path, skip_frames=2)

    # st.video(output_video_path)
    with open(output_video_path, 'rb') as processed_file:
        st.download_button(
            label="Download Processed Video",
            data=processed_file,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )
    
    
    # Display the timestamps and missing PPE details of violations
    st.subheader("PPE Violations:")
    for violation in ppe_violation_data:
        st.write(f"Timestamp: {violation['timestamp']} seconds - Missing: {', '.join(violation['missing_ppe'])}")