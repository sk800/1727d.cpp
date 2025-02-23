import cv2
from ultralytics import YOLO
import streamlit as st
import tempfile
import os
import numpy as np
import time
from moviepy import VideoFileClip, ImageSequenceClip 
import pygame

# for environments variables 
from dotenv import load_dotenv
import os
# emails 
from emails import create_email, send_email

# databases 
from database_tools import save_images_to_db


# Initialize Pygame mixer for audio
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav") # Replace "alarm.wav" with your alarm sound file

st.image("coforge logo.png", width=100) # Replace with actual path
st.title('PPE Detection App')

# Container to store timestamps and missing PPE details of violations
ppe_violation_data = []

# Define available PPE classes
available_ppe_classes = ["helmet", "jacket", "mask", "gloves", "shoes"]

# Streamlit multiselect for choosing PPE classes to detect
selected_ppe_classes = st.multiselect( "Choose PPE for detection", available_ppe_classes, default=available_ppe_classes )

# Streamlit checkbox for enabling/disabling alarm sound
play_alarm = st.checkbox("Enable alarm sound", value=True)



load_dotenv()
# Streamlit UI Inputs
from_email = os.getenv("email")
password = os.getenv("pw")
to_email = st.text_input("Recipient Email", "admin@example.com")
subject = "alert"
message = "Someone is not wearing PPE in the video. Please take action immediately."
# image_file = st.file_uploader("Upload an Image (Optional)", type=["png", "jpg", "jpeg"])

# Option to send with or without an image
send_with_image = True


def alert_notification():
    email_msg = create_email(from_email, to_email, subject, message, None, send_with_image)
    if email_msg:
        send_email(from_email, password, to_email, email_msg)
    else:
        st.error("❌ Email creation failed. Check logs.")
    

def check_ppe_compliance(results, selected_ppe_classes):
    """Checks if a person detected in a frame has all required PPE.

    Args:
        results (ultralytics.engine.results.Results): YOLOv8 detection results.
        selected_ppe_classes (list): List of PPE classes selected for detection.

    Returns:
        bool: True if compliant (person has all PPE), False otherwise.
        list: List of missing PPE classes.
    """
    detected_classes = [results.names[int(cls)] for cls in results.boxes.cls]
    if "person" in detected_classes:
        missing_ppe = [
            ppe for ppe in selected_ppe_classes if ppe not in detected_classes
        ]
        return len(missing_ppe) == 0, missing_ppe
    else:
        return True, []  # No person detected, assume compliant

def extract_snapshots(video_path, timestamps):
    """Extracts snapshots from a video at specified timestamps.

    Args:
        video_path (str): Path to the input video file.
        timestamps (list): List of timestamps (in seconds) to extract snapshots from.

    Returns:
        list: List of snapshot images (NumPy arrays).
    """
    snapshots = []
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    for timestamp in timestamps:
        frame_index = int(timestamp * fps)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = video_capture.read()
        if ret:
            snapshots.append(frame)
    video_capture.release()
    return snapshots


def process_and_display(model_path, video_path, selected_ppe_classes, skip_frames=5, play_alarm=True):
    """Processes a video with a YOLOv8 model and displays the output in Streamlit.

    Args:
        model_path (str): Path to the YOLOv8 model file (.pt).
        video_path (str): Path to the input video file.
        selected_ppe_classes (list): List of PPE classes selected for detection.
        skip_frames (int, optional): Number of frames to skip between detections.
                                    Defaults to 5 for faster processing.
        play_alarm (bool, optional): Whether to play the alarm sound for violations.
                                    Defaults to True.

    Returns:
        str: Path to the processed output video file.
        list: List of violation timestamps.
    """
    model = YOLO(model_path)
    video_clip = VideoFileClip(video_path)
    fps = video_clip.fps
    processed_frames = []
    stframe = st.empty()
    progress_bar = st.progress(0)
    total_frames = int(video_clip.fps * video_clip.duration)
    frames_processed = 0

    violation_timestamps = []  # Store violation timestamps

    for i, frame in enumerate(video_clip.iter_frames(fps=fps, dtype="uint8")):
        if i % skip_frames != 0:
            continue

        results = model(frame)

        # Access detection results
        result = results[0] if len(results) > 0 else None

        # Render bounding boxes and check for PPE compliance
        if result is not None:
            annotated_frame = result.plot()
            compliant, missing_ppe = check_ppe_compliance(
                result, selected_ppe_classes
            )
            if not compliant:
                # Display warning message on the frame
                warning_message = f"PPE Violation: Missing {', '.join(missing_ppe)}"
                cv2.putText(
                    annotated_frame,warning_message,(10, 30),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2,
                )
                
                # send notification through email
                alert_notification()
                
                # st.warning(
                #     warning_message
                # )  # Display warning in Streamlit

                # Store timestamp and missing PPE details
                timestamp = i / fps
                ppe_violation_data.append(
                    {"timestamp": timestamp, "missing_ppe": missing_ppe}
                )
                violation_timestamps.append(
                    timestamp
                )  # Store violation timestamp

                # Play alarm sound (only if enabled)
                if play_alarm:
                    alarm_sound.play()

                # Send notification
                violation_details = f"PPE Violation at timestamp {timestamp:.2f} seconds: Missing {', '.join(missing_ppe)}"
                
        else:
            annotated_frame = frame

        processed_frames.append(annotated_frame)
        stframe.image(annotated_frame, channels="RGB")

        frames_processed += skip_frames
        progress = frames_processed / total_frames
        progress_bar.progress(
            progress if progress <= 1 else 1
        )  # Ensure progress <= 1

    stframe.empty()
    progress_bar.empty()

    output_clip = ImageSequenceClip(processed_frames, fps=fps // skip_frames)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
        output_clip.write_videofile(
            temp_output.name, codec="libx264", audio=False
        )
        output_path = temp_output.name

    return output_path, violation_timestamps





# Create two tabs



model_path = "best.pt" # Update with your model path

uploaded_video = st.file_uploader( "Upload a video", type=["mp4", "avi", "mov", "mkv"] )


if uploaded_video and from_email and password and to_email:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_video.read())

    input_video_path = temp_input.name

    st.write("Processing video... This might take a while.")
    output_video_path, violation_timestamps = process_and_display(
        model_path, input_video_path, selected_ppe_classes, skip_frames=2, play_alarm=play_alarm
    )

        
        
    # st.video(output_video_path)  # Display the processed video

    # Allow users to download the processed video
    # with open(output_video_path, "rb") as processed_file:
    #     st.download_button(
    #         label="Download Processed Video",
    #         data=processed_file,
    #         file_name="processed_video.mp4",
    #         mime="video/mp4",
    #     )

    # Display the timestamps and missing PPE details of violations
    # st.subheader("PPE Violations:")
    # for violation in ppe_violation_data:
    #     st.write(
    #         f"Timestamp: {violation['timestamp']} seconds - Missing: {', '.join(violation['missing_ppe'])}"
    #     )

    # Extract and display snapshots of violations
    if violation_timestamps:
        st.subheader("Violation Snapshots:")
        snapshots = extract_snapshots(output_video_path, violation_timestamps)
            
        # saving snapshots in the database
        try:
            save_images_to_db(snapshots)
            st.success("Images saved to database")
        except Exception as e:
            print("Error occured {e}")
            
            

   
