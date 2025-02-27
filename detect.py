import torch
import cv2
import time
import json
import os
from datetime import datetime, timedelta
import requests
from ultralytics import YOLO
import pandas as pd

# Load Model
model = YOLO('./new_object_best.pt')

# Video File Path
VIDEO_PATH = "/videos/test_1_09_12_2024.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

# Hardcoded video start time
VIDEO_START_DATE = "2024-12-09"  # Format: YYYY-MM-DD
VIDEO_START_TIME = "13:20:56"    # Format: HH:MM:SS


# Load GPS data from CSV
def load_gps_data(csv_file):
    df = pd.read_csv(csv_file)
    
    # Convert dataframe to a list of dictionaries
    gps_data = df.to_dict(orient="records")  # Each row becomes a dictionary
    
    return gps_data

# Example usage
gps_data = load_gps_data("./location_datasets/Data_ride_2_location.csv")

#Convert start time to datetime object
start_datetime = datetime.strptime(f"{VIDEO_START_DATE} {VIDEO_START_TIME}", "%Y-%m-%d %H:%M:%S")

def get_gps_location(current_time):
    """Finds the closest GPS coordinate for the given timestamp."""
    for entry in gps_data:
        entry_time = datetime.strptime(f"{entry['date']} {entry['time']}", "%Y-%m-%d %H:%M:%S")
        if entry_time >= current_time:
            return entry['latitude'], entry['longitude']
    return None, None  # Return None if no match found

# Cloud Upload Settings
UPLOAD_URL = "https://your-cloud-api.com/upload"

def detect_objects(frame):
    """Run YOLOv5 Object Detection"""
    results = model(frame)
    detections = results.pandas().xyxy[0]
    detected_objects = []
    
    for _, row in detections.iterrows():
        detected_objects.append({
            "object": row['name'],
            "confidence": float(row['confidence']),
            "bbox": [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]
        })
    
    return detected_objects

def save_frame(frame, timestamp):
    """Save detected frame as an image"""
    filename = f"/detected_frames/{VIDEO_START_DATE}/frame_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    return filename

def upload_to_cloud(image_path, detected_objects, timestamp):
    """Upload image + metadata to the cloud"""
    lat, lon = get_gps_location(timestamp)
    metadata = {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "latitude": lat,
        "longitude": lon,
        "detections": detected_objects
    }
    
    with open(image_path, 'rb') as img_file:
        files = {'image': img_file}
        response = requests.post(UPLOAD_URL, files=files, data={"metadata": json.dumps(metadata)})
    
    print("Upload Response:", response.status_code, response.text)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = 1  # Process every frame

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break
    
    # Calculate current timestamp
    current_time = start_datetime + timedelta(seconds=frame_count // fps)
    
    objects = detect_objects(frame)
    if objects:
        frame_path = save_frame(frame, current_time)
        upload_to_cloud(frame_path, objects, current_time)
    
    frame_count += frame_interval
    time.sleep(1)  # Adjust as needed

cap.release()
cv2.destroyAllWindows()
