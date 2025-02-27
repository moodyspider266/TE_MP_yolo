import torch
import cv2
import time
import json
import os
import boto3
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd
from ultralytics import YOLO
import uuid

# AWS Configuration
AWS_REGION = "ap-south-1"  # Replace with your AWS region
S3_BUCKET_NAME = "te-mp-review-1"  # Replace with your bucket name
DYNAMODB_TABLE_NAME = "te-mp-image-metadata"  # Replace with your table name

# Initialize AWS services
s3_client = boto3.client('s3', region_name=AWS_REGION)
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
table = dynamodb.Table(DYNAMODB_TABLE_NAME)

# Load YOLO Model
model = YOLO('./new_object_best.pt')

# Video File Path
VIDEO_PATH = "./videos/test_1_09_12_2024.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

# Hardcoded video start time
VIDEO_START_DATE = "2024-12-09"  # Format: YYYY-MM-DD
VIDEO_START_TIME = "13:20:56"    # Format: HH:MM:SS

# Ensure local directory exists for temporary storage
LOCAL_TEMP_DIR = "/tmp/detected_frames"
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
os.makedirs(f"{LOCAL_TEMP_DIR}/{VIDEO_START_DATE}", exist_ok=True)

# Load GPS data from CSV
def load_gps_data(csv_file):
    df = pd.read_csv(csv_file)
    
    # Convert dataframe to a list of dictionaries
    gps_data = df.to_dict(orient="records")
    
    return gps_data

# Example usage
gps_data = load_gps_data("./location_datasets/Data_ride_2_location.csv")

# Convert start time to datetime object
start_datetime = datetime.strptime(f"{VIDEO_START_DATE} {VIDEO_START_TIME}", "%Y-%m-%d %H:%M:%S")

def get_gps_location(current_time):
    """Finds the closest GPS coordinate for the given timestamp."""
    closest_entry = None
    min_time_diff = timedelta.max
    
    for entry in gps_data:
        entry_time = datetime.strptime(f"{entry['date']} {entry['time']}", "%Y-%m-%d %H:%M:%S")
        time_diff = abs(entry_time - current_time)
        
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_entry = entry
    
    if closest_entry:
        # Convert floats to Decimal
        lat = Decimal(str(closest_entry['latitude'])) if isinstance(closest_entry['latitude'], float) else closest_entry['latitude']
        lon = Decimal(str(closest_entry['longitude'])) if isinstance(closest_entry['longitude'], float) else closest_entry['longitude']
        return lat, lon
    return None, None  # Return None if no match found

def detect_objects(frame):
    """Run YOLO Object Detection"""
    results = model(frame)
    
    # Handle updated return format of YOLO (may vary based on your ultralytics version)
    if hasattr(results, 'pandas'):
        # For older versions
        detections = results.pandas().xyxy[0]
        detected_objects = []
        
        for _, row in detections.iterrows():
            detected_objects.append({
                "object": row['name'],
                "confidence": float(row['confidence']),
                "bbox": [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]
            })
    else:
        # For newer versions
        detected_objects = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                detected_objects.append({
                    "object": class_name,
                    "confidence": confidence,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
    
    return detected_objects

def save_frame_locally(frame, timestamp):
    """Save detected frame as an image locally"""
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{LOCAL_TEMP_DIR}/{VIDEO_START_DATE}/frame_{timestamp_str}.jpg"
    cv2.imwrite(filename, frame)
    return filename

def upload_to_s3(local_path, timestamp):
    """Upload image to S3 bucket"""
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
    s3_key = f"detected_frames/{VIDEO_START_DATE}/frame_{timestamp_str}.jpg"
    
    try:
        s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_key)
        s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
        return s3_url
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return None

def save_to_dynamodb(detection_id, timestamp, s3_url, detected_objects, lat, lon):
    """Save metadata to DynamoDB"""
    try:
        # Convert all float values to Decimal for DynamoDB
        decimal_objects = []
        for obj in detected_objects:
            decimal_obj = {
                "object": obj["object"],
                "confidence": Decimal(str(obj["confidence"])),
                "bbox": [Decimal(str(i)) if isinstance(i, float) else i for i in obj["bbox"]]
            }
            decimal_objects.append(decimal_obj)
        
        # Prepare item with Decimal values
        item = {
            'image_id': detection_id,
            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
            'date': timestamp.strftime("%Y-%m-%d"),
            'time': timestamp.strftime("%H:%M:%S.%f"),
            'image_url': s3_url
        }
        
        # Add lat/lon if available
        if lat is not None:
            item['latitude'] = lat
        else:
            item['latitude'] = 'unknown'
            
        if lon is not None:
            item['longitude'] = lon
        else:
            item['longitude'] = 'unknown'
            
        # Add detections as a JSON string with Decimal values
        item['detections'] = decimal_objects
        
        table.put_item(Item=item)
        return True
    except Exception as e:
        print(f"Error saving to DynamoDB: {e}")
        print(f"Error details: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def process_detection(frame, current_time, detected_objects):
    """Process a detection by saving image and metadata"""
    # Generate a unique ID for this detection
    detection_id = str(uuid.uuid4())
    
    # Save frame locally
    local_path = save_frame_locally(frame, current_time)
    
    # Get GPS coordinates
    lat, lon = get_gps_location(current_time)
    
    # Upload to S3
    s3_url = upload_to_s3(local_path, current_time)
    
    if s3_url:
        # Save metadata to DynamoDB
        success = save_to_dynamodb(detection_id, current_time, s3_url, detected_objects, lat, lon)
        
        if success:
            print(f"Successfully processed detection at {current_time}")
        else:
            print(f"Failed to save metadata to DynamoDB for detection at {current_time}")
    else:
        print(f"Failed to upload image to S3 for detection at {current_time}")
    
    # Optionally clean up local file after upload
    if os.path.exists(local_path):
        os.remove(local_path)

def process_video():
    """Main function to process video file"""
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Processing video with {total_frames} frames at {fps} FPS")
    
    frame_count = 0
    detection_count = 0
    
    # Calculate expected processing time
    estimated_time_per_frame = 0.1  # seconds, will be updated during processing
    estimated_total_time = total_frames * estimated_time_per_frame
    print(f"Estimated processing time: {estimated_total_time/60:.1f} minutes")
    
    start_time = time.time()
    last_progress_time = start_time
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break
            
            frame_start_time = time.time()
            
            # Calculate current timestamp
            current_time = start_datetime + timedelta(seconds=frame_count / fps)
            
            # Detect objects in the frame
            objects = detect_objects(frame)
            # After: objects = detect_objects(frame)
            print(f"Frame {frame_count}: Detected {len(objects)} objects")
            if objects:
                print(f"Objects detected: {[obj['object'] for obj in objects]}")
            
            # If objects are detected, process the detection
            if objects:
                process_detection(frame, current_time, objects)
                detection_count += 1
            
            frame_count += 1
            
            # Update estimated time per frame
            frame_end_time = time.time()
            frame_processing_time = frame_end_time - frame_start_time
            estimated_time_per_frame = (estimated_time_per_frame * (frame_count - 1) + frame_processing_time) / frame_count
            
            # Show progress every 5 seconds
            if time.time() - last_progress_time > 5:
                elapsed_time = time.time() - start_time
                progress = frame_count / total_frames
                estimated_remaining = (total_frames - frame_count) * estimated_time_per_frame
                
                print(f"Progress: {progress*100:.1f}% ({frame_count}/{total_frames} frames)")
                print(f"Detections: {detection_count}")
                print(f"Elapsed time: {elapsed_time/60:.1f} minutes")
                print(f"Estimated remaining time: {estimated_remaining/60:.1f} minutes")
                print(f"Average processing time per frame: {estimated_time_per_frame:.3f} seconds")
                print("-" * 50)
                
                last_progress_time = time.time()
    finally:
        # Release resources safely
        cap.release()
        
        # Only try to destroy windows if running in GUI mode
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # Ignore OpenCV window errors
            pass
        
        # Print final stats
        total_time = time.time() - start_time
        print(f"Video processing complete in {total_time/60:.1f} minutes.")
        print(f"Processed {frame_count} frames with {detection_count} detections.")
        if frame_count > 0:
            print(f"Average time per frame: {total_time/frame_count:.3f} seconds")

if __name__ == "__main__":
    process_video()