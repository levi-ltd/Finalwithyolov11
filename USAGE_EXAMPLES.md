# Singer Detection System - Usage Examples

This document provides detailed examples of how to use the YOLOv11 Singer Detection System in various scenarios.

## 🎤 Basic Singer Detection

### Quick Test with Demo

```powershell
# Test the system with built-in demo
python demo.py check          # Verify installation
python demo.py camera         # Live camera demo
python demo.py video sample_performance.mp4  # Process video file
```

### Real-time Detection

```powershell
# Start live singer detection
python src/main.py --source camera --device 0

# With video recording
python src/main.py --source camera --device 0 --output results/live_performance.mp4

# Using specific camera (if multiple available)
python src/main.py --source camera --device 1
```

**What you'll see:**
- 🔴 Red boxes: People without microphones
- 🔵 Blue boxes: Microphones
- 🟢 Green boxes: Singers (people + microphone)
- 🎵 Music note symbol on singer boxes
- Distance number showing microphone proximity

## 🎬 Video Processing Examples

### Concert Recording Analysis

```powershell
# Process a full concert recording
python src/main.py --source video --input concert_full_show.mp4 --output results/concert_analysis.mp4

# Process specific segment (using video editing tools first)
python src/main.py --source video --input song_performance.mp4 --output results/song_analysis.mp4
```

### Karaoke Session Tracking

```powershell
# Analyze karaoke performance
python src/main.py --source video --input karaoke_night.mp4 --output results/karaoke_tracking.mp4
```

**Use case**: Track multiple singers taking turns, measure engagement time

### Live Streaming Integration

```powershell
# Start API for real-time integration
python src/api/app.py

# Test with image upload
curl -X POST "http://localhost:8000/detect" -F "file=@performance_photo.jpg"

# Get video stream processing
curl -X POST "http://localhost:8000/track" -F "file=@frame.jpg"
```

**Integration example** (JavaScript):
```javascript
// Upload frame for singer detection
const formData = new FormData();
formData.append('file', frameBlob);

fetch('http://localhost:8000/detect', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    // data.detections contains singer information
    data.detections.forEach(detection => {
        if (detection.class_name === 'singer') {
            console.log(`Singer detected! Confidence: ${detection.confidence}`);
            console.log(`Microphone distance: ${detection.micro_distance}px`);
        }
    });
});
```

## ⚙️ Configuration Examples

### Adjust Singer Sensitivity

```yaml
# config/config.yaml
singer_detection:
  proximity_threshold: 30    # Stricter detection (closer required)
  enabled: true

# For large stages, increase threshold
singer_detection:
  proximity_threshold: 100   # Looser detection (farther allowed)
```

### Optimize for Performance Type

**Close-up performances (karaoke, small venues):**
```yaml
singer_detection:
  proximity_threshold: 25

model:
  conf_threshold: 0.30
  name: "yolo11s"           # Good balance of speed/accuracy

camera:
  width: 640
  height: 480
  fps: 30
```

**Large stage performances:**
```yaml
singer_detection:
  proximity_threshold: 75

model:
  conf_threshold: 0.25
  name: "yolo11m"           # Higher accuracy for distant objects

camera:
  width: 1920
  height: 1080
  fps: 25
```

### Low-resource devices:**
```yaml
singer_detection:
  proximity_threshold: 50

model:
  conf_threshold: 0.35
  name: "yolo11n"           # Fastest model

camera:
  width: 480
  height: 360
  fps: 20
```

## 📊 Data Collection & Training

### Collect Performance Data

```powershell
# Record performance sessions for training data
python src/main.py --source camera --device 0 --output training_data/session_001.mp4

# Process existing videos to extract frames
python scripts/extract_frames.py --input concert_video.mp4 --output training_data/frames/
```

### Label with Label Studio

```powershell
# Start Label Studio
python src/labeling/start_labelstudio.py

# Or start manually
label-studio start --port 8080
```

**Labeling workflow:**
1. Upload performance images/videos to Label Studio
2. Draw boxes around:
   - **Person**: Anyone without a microphone
   - **Micro**: All visible microphones 
   - **Singer**: People actively holding/using microphones
3. Export labeled data for training

### Train Custom Model

```powershell
# Prepare your dataset
python src/training/train.py prepare --data-dir training_data/labeled

# Train model
python src/training/train.py train --data training_data/dataset.yaml --epochs 50 --batch-size 8

# Validate trained model
python src/training/train.py validate --model models/trained/best.pt --data training_data/dataset.yaml
```

## 🎭 Advanced Use Cases

### Theater Production Monitoring

```python
# Custom script for theater applications
import cv2
from src.detection.detector import YOLODetector
from src.tracking.tracker import ObjectTracker
from src.utils.config import load_config

# Load configuration
config = load_config('config/config.yaml')

# Initialize components
detector = YOLODetector(config)
tracker = ObjectTracker(config)

# Process theater performance
cap = cv2.VideoCapture('theater_performance.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect and track
    detections = detector.detect(frame)
    tracked_objects = tracker.update(detections)
    
    # Analyze speaking actors
    speaking_actors = [obj for obj in tracked_objects if obj.class_name == 'singer']
    print(f"Speaking actors: {len(speaking_actors)}")
    
    # Add your custom logic here
    # - Track dialogue timing
    # - Monitor stage positioning
    # - Analyze performance flow

cap.release()
```

### Multi-camera Setup

```python
# Process multiple camera angles simultaneously
import threading
from src.camera.camera_manager import CameraManager

def process_camera(camera_id, output_path):
    # Initialize camera
    camera_manager = CameraManager(config, device=camera_id)
    
    # Process frames
    while True:
        frame = camera_manager.get_frame()
        if frame is None:
            break
            
        # Detect singers
        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections)
        
        # Save results
        # ... processing logic

# Start multiple camera threads
cameras = [0, 1, 2]  # Camera indices
threads = []

for cam_id in cameras:
    thread = threading.Thread(target=process_camera, args=(cam_id, f'output_cam_{cam_id}.mp4'))
    thread.start()
    threads.append(thread)

# Wait for completion
for thread in threads:
    thread.join()
```

### Performance Analytics

```python
# Generate performance statistics
import pandas as pd
import matplotlib.pyplot as plt

def analyze_performance(tracking_data):
    """Generate performance analytics from tracking data"""
    df = pd.DataFrame(tracking_data)
    
    # Singer engagement analysis
    singer_data = df[df['class_name'] == 'singer']
    
    # Calculate singer activity
    singer_activity = singer_data.groupby('track_id').agg({
        'frame': ['min', 'max', 'count'],
        'confidence': 'mean',
        'micro_distance': 'mean'
    }).round(2)
    
    print("Singer Activity Summary:")
    print(singer_activity)
    
    # Plot singer detection over time
    plt.figure(figsize=(12, 6))
    singers_per_frame = singer_data.groupby('frame').size()
    plt.plot(singers_per_frame.index, singers_per_frame.values)
    plt.title('Singer Activity Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Number of Active Singers')
    plt.grid(True)
    plt.savefig('singer_activity.png')
    plt.show()
    
    return singer_activity

# Use after processing video
# tracking_data = process_video('performance.mp4')
# analytics = analyze_performance(tracking_data)
```

## 🔧 Troubleshooting Examples

### Debug Detection Issues

```python
# Debug script to test detection on sample images
import cv2
import matplotlib.pyplot as plt
from src.detection.detector import YOLODetector
from src.utils.config import load_config

# Load test image
image = cv2.imread('test_performance.jpg')
config = load_config('config/config.yaml')
detector = YOLODetector(config)

# Run detection
detections = detector.detect(image)

# Print results
print(f"Found {len(detections)} objects:")
for i, det in enumerate(detections):
    print(f"  {i+1}. {det.class_name} (confidence: {det.confidence:.2f})")
    if hasattr(det, 'has_micro'):
        print(f"     Has microphone: {det.has_micro}")
        print(f"     Microphone distance: {det.micro_distance:.1f}px")

# Visualize
from src.utils.visualizer import Visualizer
visualizer = Visualizer(config)
annotated_image = visualizer.draw_detections(image, detections)

plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.title('Singer Detection Results')
plt.axis('off')
plt.show()
```

### Optimize Proximity Threshold

```python
# Script to find optimal proximity threshold
import numpy as np

def test_proximity_thresholds(image_path, ground_truth_singers):
    """Test different proximity thresholds to find optimal value"""
    
    thresholds = [20, 30, 40, 50, 60, 70, 80]
    results = []
    
    for threshold in thresholds:
        # Update config
        config['singer_detection']['proximity_threshold'] = threshold
        detector = YOLODetector(config)
        
        # Run detection
        image = cv2.imread(image_path)
        detections = detector.detect(image)
        
        # Count detected singers
        detected_singers = len([d for d in detections if d.class_name == 'singer'])
        
        # Calculate accuracy (you need to provide ground truth)
        accuracy = calculate_accuracy(detected_singers, ground_truth_singers)
        
        results.append({
            'threshold': threshold,
            'detected': detected_singers,
            'accuracy': accuracy
        })
        
        print(f"Threshold {threshold}: {detected_singers} singers, {accuracy:.2%} accuracy")
    
    return results

# results = test_proximity_thresholds('concert_sample.jpg', ground_truth=3)
```

## 📱 Mobile/Web Integration

### Flask Web Application

```python
# Simple web interface for singer detection
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from src.detection.detector import YOLODetector
from src.utils.config import load_config

app = Flask(__name__)

# Initialize detector
config = load_config('config/config.yaml')
detector = YOLODetector(config)

@app.route('/')
def index():
    return '''
    <html>
    <body>
        <h2>Singer Detection System</h2>
        <form method="POST" action="/detect" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">Detect Singers</button>
        </form>
    </body>
    </html>
    '''

@app.route('/detect', methods=['POST'])
def detect_singers():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})
    
    file = request.files['image']
    
    # Read image
    image_data = file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Detect singers
    detections = detector.detect(image)
    
    # Format results
    results = []
    for det in detections:
        result = {
            'class': det.class_name,
            'confidence': float(det.confidence),
            'bbox': det.bbox.tolist()
        }
        if hasattr(det, 'has_micro'):
            result['has_microphone'] = det.has_micro
            result['microphone_distance'] = float(det.micro_distance)
        results.append(result)
    
    return jsonify({
        'total_objects': len(detections),
        'singers': len([d for d in detections if d.class_name == 'singer']),
        'detections': results
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### Real-time Streaming

```python
# Stream processing with WebSocket
import asyncio
import websockets
import cv2
import json
import base64

async def process_stream(websocket, path):
    """Process real-time video stream"""
    
    detector = YOLODetector(config)
    
    async for message in websocket:
        try:
            # Decode frame from client
            data = json.loads(message)
            image_data = base64.b64decode(data['frame'])
            
            # Convert to opencv format
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process frame
            detections = detector.detect(frame)
            
            # Send results back
            results = {
                'singers': len([d for d in detections if d.class_name == 'singer']),
                'timestamp': data.get('timestamp', 0),
                'detections': [
                    {
                        'class': d.class_name,
                        'confidence': float(d.confidence),
                        'has_microphone': getattr(d, 'has_micro', False)
                    } for d in detections
                ]
            }
            
            await websocket.send(json.dumps(results))
            
        except Exception as e:
            await websocket.send(json.dumps({'error': str(e)}))

# Start WebSocket server
start_server = websockets.serve(process_stream, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
```

This comprehensive usage guide provides examples for virtually every use case, from basic detection to advanced integrations. Users can pick the examples that match their specific needs and build upon them.