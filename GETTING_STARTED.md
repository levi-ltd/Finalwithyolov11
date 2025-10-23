# Getting Started with YOLOv11 Object Detection and Tracking

This guide will help you get up and running with the YOLOv11 object detection and tracking system in just a few minutes.

## Prerequisites

- Windows 10/11 with PowerShell
- Python 3.8 or higher
- Webcam (optional, for real-time detection)
- Git (to clone additional resources if needed)

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

### Real-time Camera Detection

```powershell
# Basic camera detection
python src\main.py --source camera --device 0

# With custom settings
python src\main.py --source camera --device 0 --output results\camera_output.mp4
```

### Video File Processing

```powershell
# Process a video file
python src\main.py --source video --input path\to\video.mp4

# Save results
python src\main.py --source video --input video.mp4 --output results\processed_video.mp4
```

### Start Label Studio for Annotation

```powershell
# Start Label Studio server
python src\labeling\start_labelstudio.py
```

Then open http://localhost:8080 in your browser to access the annotation interface.

### API Server

```powershell
# Start the REST API server
python src\api\app.py
```

The API will be available at http://localhost:8000

Test the API:
```powershell
# Using curl (if available)
curl -X POST "http://localhost:8000/detect" -F "file=@image.jpg"
```

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

### Camera Not Working
```powershell
# Check available cameras
python -c "import cv2; print('Camera available:', cv2.VideoCapture(0).isOpened())"

# Try different camera indices (0, 1, 2, etc.)
python src\main.py --source camera --device 1
```

### Low Performance
- Use a smaller model: Change `yolo11n` to `yolo11s` in config
- Reduce image size in camera settings
- Close other applications using the camera

### Import Errors
```powershell
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print(sys.path)"
```

### Label Studio Not Starting
```powershell
# Install Label Studio separately
pip install label-studio

# Start manually
label-studio start --port 8080
```

## Next Steps

1. **Customize Detection Classes**
   - Edit `config\dataset.yaml` to define your object classes
   - Update the Label Studio configuration for your specific use case

2. **Train Custom Models**
   - Collect and label your own data using Label Studio
   - Use the training utilities in `src\training\`

3. **Integrate with Your Application**
   - Use the REST API for web integration
   - Import the modules directly for custom Python applications

4. **Optimize Performance**
   - Experiment with different model sizes
   - Tune tracking parameters for your specific scenario
   - Consider GPU acceleration for better performance

## Directory Structure

```
ObjectDectionYolo11/
├── src/                    # Source code
│   ├── detection/         # YOLO detection
│   ├── tracking/          # Multi-object tracking  
│   ├── camera/            # Camera management
│   ├── labeling/          # Label Studio integration
│   ├── training/          # Model training
│   ├── api/               # REST API
│   └── utils/             # Utilities
├── config/                # Configuration files
├── data/                  # Dataset storage
├── models/                # Model weights
├── results/               # Output results
├── notebooks/             # Jupyter tutorials
├── requirements.txt       # Dependencies
├── setup.py              # Setup script
└── demo.py               # Quick demo
```

## Support

- Check the `DEVELOPMENT.md` file for technical details
- Review the Jupyter notebook for comprehensive examples
- Examine the configuration files in `config/`
- Look at the source code in `src/` for implementation details

Happy tracking! 🎯