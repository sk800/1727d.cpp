import streamlit as st
import cv2
import numpy as np
import tempfile
import pygame
import psycopg2
import json
import binascii
from ultralytics import YOLO
from moviepy import VideoFileClip
from collections import defaultdict
from emails import create_email, send_email  # Import email functionality
from tracker import Tracker  # Import tracking logic

# Initialize pygame mixer for alarm sound
pygame.mixer.init()
alarm_sound = "alarm.wav"

def get_connection():
    try:
        conn = psycopg2.connect(
            host="localhost", database="image_store", user="postgre", password="cfg@1234"
        )
        return conn
    except psycopg2.OperationalError as e:
        print("Error: Unable to connect to the database.", e)
        return None

def save_defaulter_image(image):
    try:
        conn = get_connection()
        if conn is None:
            return
        cursor = conn.cursor()
        binary_image = image.tobytes()
        binary_image_hex = binascii.hexlify(binary_image).decode()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS defaulters (
                id SERIAL PRIMARY KEY,
                image BYTEA NOT NULL
            )""")
        cursor.execute("INSERT INTO defaulters (image) VALUES (%s)", (binary_image_hex,))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("Error saving defaulter image:", e)

def retrieve_defaulter_images():
    try:
        conn = get_connection()
        if conn is None:
            return []
        cursor = conn.cursor()
        cursor.execute("SELECT image FROM defaulters")
        results = cursor.fetchall()
        images = [binascii.unhexlify(row[0]) for row in results]
        cursor.close()
        conn.close()
        return images
    except Exception as e:
        print("Error retrieving defaulter images:", e)
        return []

st.sidebar.title("Navigation")
st.image("coforge logo.png", width=100)
st.sidebar.image("coforge logo.png", width=100)
app_choice = st.sidebar.radio("Choose an app", ["People Entry Counter", "PPE Detection App", "Defaulter Images"])

if app_choice == "PPE Detection App":
    st.title("PPE Detection App")
    model_path = 'best.pt'
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    available_ppe = ["helmet", "jacket", "mask", "gloves", "safety glass", "shoes"]
    selected_ppe = st.multiselect("Select PPE to detect", available_ppe, default=available_ppe)
    admin_email = st.text_input("Enter Admin Email", "admin@example.com")
    trigger_alarm = st.checkbox("Trigger Alarm on PPE Violation", value=True)
    captured_ids = set()
    
    def check_ppe_compliance(results):
        detected_classes = [results.names[int(cls)] for cls in results.boxes.cls]
        if "person" in detected_classes:
            missing_ppe = [ppe for ppe in selected_ppe if ppe not in detected_classes]
            return len(missing_ppe) == 0, missing_ppe
        return True, []
    
    def process_and_display(model_path, video_path, skip_frames=5):
        model = YOLO(model_path)
        video_clip = VideoFileClip(video_path)
        fps = video_clip.fps
        stframe = st.empty()
        progress_bar = st.progress(0)
        total_frames = int(video_clip.fps * video_clip.duration)
        frames_processed = 0
        tracker = model.track
        for i, frame in enumerate(video_clip.iter_frames(fps=fps, dtype="uint8")):
            if i % skip_frames != 0:
                continue
            results = tracker(frame, persist=True)
            result = results[0] if len(results) > 0 else None
            if result is not None:
                annotated_frame = result.plot()
                for j, (cls, box) in enumerate(zip(result.boxes.cls, result.boxes.xywh)):
                    if int(cls) == 0:
                        if result.boxes.id is not None:
                            track_id = result.boxes.id[j].item()
                            compliant, missing_ppe = check_ppe_compliance(result)
                            if not compliant and track_id not in captured_ids:
                                captured_ids.add(track_id)
                                _, snapshot_path = tempfile.mkstemp(suffix=".jpg")
                                cv2.imwrite(snapshot_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                                save_defaulter_image(annotated_frame)
                                if trigger_alarm:
                                    pygame.mixer.music.load(alarm_sound)
                                    pygame.mixer.music.play()
            else:
                annotated_frame = frame
            stframe.image(annotated_frame, channels="RGB")
            frames_processed += skip_frames
            progress_bar.progress(min(frames_processed / total_frames, 1))
        stframe.empty()
        progress_bar.empty()
    
    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(uploaded_video.read())
            input_video_path = temp_input.name
        st.write("Processing video... This might take a while.")
        process_and_display(model_path, input_video_path, skip_frames=2)

elif app_choice == "Defaulter Images":
    st.title("Defaulter Images")
    if st.button("Fetch Defaulter Images"):
        images = retrieve_defaulter_images()
        if images:
            for img in images:
                np_img = np.frombuffer(img, dtype=np.uint8)
                np_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                st.image(np_img, caption="Defaulter Image", channels="BGR")
        else:
            st.write("No defaulter images found.")
