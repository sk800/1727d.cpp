from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
 
# Load YOLO model
model = YOLO("yolov8n.pt")
 
# Initialize DeepSORT with tuned parameters
tracker = DeepSort(
    max_age=70,               # Keep IDs longer during occlusions
    # min_confidence=0.3,       # Lower threshold for detections
    nms_max_overlap=0.7,      # Reduce duplicate detections
    embedder="mobilenet",     # Use appearance features for ID consistency
)
 
# Open video
video_path = "input_video.mp4"
cap = cv2.VideoCapture(0)
 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 
    # Run YOLO object detection
    results = model(frame)
    detections = []
 
    # Extract bounding boxes and confidence scores
    for result in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = result.cpu().numpy()
        if int(cls) == 0:  # Only track persons (YOLO class 0)
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, int(cls), None))
 
    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)
 
    # Draw tracks
    for track in tracks:
        if track.is_confirmed() and track.time_since_update == 0:
            bbox = track.to_tlbr()
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
 
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
 
cap.release()
cv2.destroyAllWindows()