# YOLOv11 Singer Detection and Tracking System

A simplified object detection and tracking system using YOLOv11 specifically designed to detect people, microphones, and automatically identify singers. When a person is detected near a microphone, they are automatically classified as a singer.

## 🎤 Key Features

- **Simplified 3-Class Detection**: Person, Microphone, Singer
- **Automatic Singer Detection**: Person + Microphone = Singer
- **Real-time Tracking**: Maintains singer identities across frames
- **Smart Proximity Analysis**: Configurable distance thresholds
- **Visual Feedback**: Color-coded detection with distance indicators
- **Label Studio Integration**: Streamlined annotation workflow
- **REST API**: Easy integration for live performance systems

## Project Structure

```
ObjectDectionYolo11/
├── src/
│   ├── detection/          # YOLOv11 detection modules
│   ├── tracking/           # Multi-object tracking
│   ├── camera/             # Camera input handling
│   ├── labeling/           # Label Studio integration
│   ├── training/           # Model training utilities
│   ├── api/                # REST API endpoints
│   └── utils/              # Utility functions
├── config/                 # Configuration files
├── data/                   # Dataset storage
│   ├── raw/                # Raw images/videos
│   ├── labeled/            # Labeled data
│   └── processed/          # Processed datasets
├── models/                 # Trained models
├── outputs/                # Detection results
├── notebooks/              # Jupyter notebooks
└── tests/                  # Unit tests
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Real-time Singer Detection
```bash
python src/main.py --source camera --device 0
```
**What you'll see**: Red boxes for people, blue boxes for microphones, green boxes for singers

### 2. Process Performance Video
```bash
python src/main.py --source video --input performance_video.mp4
```
**Perfect for**: Concert recordings, live performances, karaoke sessions

### 3. Start Annotation Tool
```bash
python src/labeling/start_labelstudio.py
```
**Use for**: Creating training data with person/microphone/singer labels

### 4. Train Singer Detection Model
```bash
python src/training/train.py --data config/dataset.yaml --epochs 100
```
**Result**: Custom model trained specifically on your performance data

### 5. API for Live Applications
```bash
python src/api/app.py
```
**Integration**: Real-time singer detection for live streaming, performance analysis

## Configuration

Edit `config/config.yaml` to customize singer detection:

```yaml
# Singer Detection Settings
singer_detection:
  proximity_threshold: 50  # pixels for person-microphone detection
  enabled: true

# Model parameters
model:
  name: "yolo11n"  # Faster models for real-time performance
  conf_threshold: 0.25
  iou_threshold: 0.45

# Camera settings optimized for performances
camera:
  width: 1280
  height: 720
  fps: 30

# Tracking for consistent singer IDs
tracking:
  enabled: true
  max_age: 30
  min_hits: 3
```

## Singer Detection Logic

The system automatically identifies singers using proximity analysis:

1. **Detect Objects**: YOLO identifies persons and microphones
2. **Calculate Distances**: Measures pixel distance between person and microphone centers
3. **Apply Threshold**: If distance < 50 pixels (configurable), person becomes singer
4. **Track Over Time**: Maintains singer IDs as they move around stage
5. **Visual Feedback**: Green boxes for singers, red for persons, blue for microphones

## Label Studio Integration

The system includes seamless Label Studio integration for:
- Automatic data import from detection results
- Custom labeling interface for object detection
- Export labeled data for training
- Model validation with human annotations

## Usage Examples

See the `notebooks/` directory for detailed examples and tutorials.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

MIT License