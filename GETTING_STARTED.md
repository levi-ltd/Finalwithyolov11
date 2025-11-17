# Getting Started with YOLOv11 Singer Detection System

This guide will help you get up and running with the YOLOv11 singer detection system in just a few minutes. The system automatically detects people with microphones and classifies them as singers.

## 🎤 What This System Does

- **Detects People**: Identifies individuals in the frame (red boxes)
- **Finds Microphones**: Locates microphones and similar objects (blue boxes)  
- **Identifies Singers**: When a person is near a microphone, they become a singer (green boxes)
- **Tracks Over Time**: Maintains singer IDs as they move around
- **Shows Distance**: Displays microphone proximity for each detected singer

## Prerequisites

- Windows 10/11 with PowerShell
- Python 3.8 or higher
- Webcam (optional, for live testing)
- Performance videos or images for testing

## Quick Setup

### 1. Install Dependencies

Open PowerShell in the project directory and run:

```powershell
pip install -r requirements.txt
```

This will install:
- ultralytics (YOLOv11)
- opencv-python (computer vision)
- label-studio (annotation tool)
- Other required packages

### 2. Initialize the Project

Run the setup script:

```powershell
python setup.py
```

This will:
- ✅ Check Python version
- 📁 Create necessary directories  
- 📦 Install dependencies
- 🤖 Download YOLO model weights
- 📹 Test camera availability

### 3. Quick Test

Test the system with a simple demo:

```powershell
# Check if everything is set up correctly
python demo.py check

# Test with camera (if available)
python demo.py camera

# Test with a video file
python demo.py video path\to\your\video.mp4
```

## Usage Examples

### Live Singer Detection

```powershell
# Start real-time singer detection
python src\main.py --source camera --device 0
```

**What you'll see:**
- 🔴 Red boxes: People without microphones
- 🔵 Blue boxes: Microphones
- 🟢 Green boxes: Singers (people with microphones)
- 🎵 Music note symbol on singers
- Distance indicator showing microphone proximity

### Process Performance Videos

```powershell
# Analyze a concert or performance video
python src\main.py --source video --input concert_video.mp4 --output results\singer_analysis.mp4
```

**Perfect for:**
- Concert recordings analysis
- Live performance monitoring
- Karaoke session tracking
- Stage performance analytics

### Configure Singer Detection

Edit `config\config.yaml` to adjust sensitivity:

```yaml
singer_detection:
  proximity_threshold: 50    # pixels (lower = closer required)
  enabled: true

model:
  conf_threshold: 0.25      # detection confidence
  name: "yolo11n"           # model size (n=fastest, x=most accurate)
```

### Label Your Own Performance Data

```powershell
# Start Label Studio for creating training data
python src\labeling\start_labelstudio.py
```

1. Open http://localhost:8080
2. Upload performance images/videos
3. Label people, microphones, and singers
4. Export data for custom training

### API for Live Applications

```powershell
# Start API server
python src\api\app.py
```

**Test the API:**
```powershell
# Upload an image for singer detection
curl -X POST "http://localhost:8000/detect" -F "file=@performance_image.jpg"
```

**Response includes:**
- Singer detections with confidence scores
- Microphone distances for each singer
- Bounding box coordinates
- Track IDs for video sequences

## Configuration

Edit `config\config.yaml` to customize:

```yaml
# Model settings
model:
  name: "yolo11n"  # Options: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
  conf_threshold: 0.25
  iou_threshold: 0.45

# Camera settings  
camera:
  source: 0  # Camera index
  width: 1280
  height: 720
  fps: 30

# Tracking settings
tracking:
  enabled: true
  max_age: 30
  min_hits: 3
```

## Jupyter Notebook Tutorial

For a comprehensive tutorial, open the Jupyter notebook:

```powershell
# Install Jupyter if not already installed
pip install jupyter

# Start Jupyter
jupyter notebook

# Open the tutorial
# Navigate to: notebooks/yolo11_detection_tracking_tutorial.ipynb
```

The notebook covers:
- 📦 Installation and setup
- 🤖 Model configuration
- 🏷️ Label Studio integration
- 📹 Camera processing
- 📊 Results analysis
- 💾 Data export

## Common Issues & Solutions

### Singer Not Detected
```yaml
# In config\config.yaml, try adjusting:
singer_detection:
  proximity_threshold: 75  # Increase if singers not detected
  enabled: true

model:
  conf_threshold: 0.20     # Lower for more sensitive detection
```

### False Singer Detections
```yaml
# In config\config.yaml:
singer_detection:
  proximity_threshold: 30  # Decrease for stricter detection
  
model:
  conf_threshold: 0.30     # Higher for more confident detections
```

### Performance Issues
- **Use smaller model**: Change to `yolo11n` for speed
- **Reduce resolution**: Lower camera width/height in config
- **Check lighting**: Ensure adequate stage/room lighting
- **Close other apps**: Free up system resources

### Microphones Not Detected
```powershell
# The system detects these as microphones:
# - Handheld microphones
# - Cell phones (often held like mics)
# - Small handheld objects

# For better microphone detection:
# - Ensure good lighting on microphones
# - Use contrasting backgrounds
# - Consider training custom model with your specific microphones
```

### Camera Issues
```powershell
# Test camera availability
python -c "import cv2; print('Camera:', cv2.VideoCapture(0).isOpened())"

# Try different camera indices
python src\main.py --source camera --device 1
python src\main.py --source camera --device 2
```

### Training Custom Models
```powershell
# If default detection isn't working well:
# 1. Collect your performance videos/images
# 2. Use Label Studio to label people, microphones, singers
# 3. Train custom model:
python src\training\train.py --data your_dataset.yaml --epochs 50
```

## Next Steps

1. **Optimize for Your Performances**
   - Test with your typical performance setup
   - Adjust proximity threshold for your stage size
   - Train on your specific lighting conditions

2. **Create Custom Training Data**
   - Record sample performances with various microphone types
   - Use Label Studio to label people, microphones, and singers
   - Train models specific to your performance style

3. **Integrate with Performance Systems**
   - Use REST API for live streaming overlays
   - Connect to lighting systems for singer-following spots
   - Integrate with audio systems for automatic mixing

4. **Advanced Analytics**
   - Track singer movement patterns
   - Analyze performance engagement
   - Generate automated performance reports
   - Monitor microphone usage statistics

## Performance Use Cases

### 🎤 **Live Concerts**
- Track lead singers vs backup singers
- Monitor microphone handoffs
- Generate performer analytics
- Create automated camera switching

### 🎵 **Karaoke Venues**
- Automatic singer detection for scoring
- Queue management based on detected singers
- Performance recording with singer identification

### 🎭 **Theater Productions**
- Track speaking actors vs ensemble
- Monitor wireless microphone usage
- Analyze stage positioning

### 📺 **Live Streaming**
- Automatic singer highlighting
- Dynamic overlay positioning
- Performance statistics display

## Directory Structure

```
ObjectDectionYolo11/
├── src/                           # Source code
│   ├── detection/                 # Singer detection logic
│   │   └── detector.py           # 3-class detection + singer proximity
│   ├── tracking/                  # Singer tracking over time
│   │   └── tracker.py            # Maintains singer IDs
│   ├── camera/                    # Live performance capture
│   ├── labeling/                  # Annotation for person/micro/singer
│   ├── training/                  # Custom model training
│   ├── api/                       # REST API for live integration
│   └── utils/                     # Performance visualization
├── config/
│   ├── config.yaml               # Singer detection settings
│   └── dataset.yaml              # 3-class training config
├── data/                          # Performance datasets
├── models/                        # Trained singer detection models
├── results/                       # Singer analysis outputs
├── notebooks/
│   └── yolo11_detection_tracking_tutorial.ipynb  # Singer detection demo
├── requirements.txt               # Dependencies
├── setup.py                      # Setup script
└── demo.py                       # Quick singer detection test
```

## Support & Resources

- **Technical Details**: Check `DEVELOPMENT.md` for architecture info
- **Complete Tutorial**: Run the Jupyter notebook for step-by-step examples
- **Configuration**: Review `config/config.yaml` for all settings
- **Source Code**: Examine `src/` directory for implementation details
- **Performance Tips**: See troubleshooting section above

**Happy singer tracking! 🎤🎵**