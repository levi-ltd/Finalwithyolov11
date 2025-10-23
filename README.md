# YOLOv11 Object Detection and Tracking System

A comprehensive object detection and tracking system using YOLOv11 with Label Studio integration for data annotation and model training.

## Features

- Real-time object detection using YOLOv11
- Multi-object tracking with Kalman filters
- Camera input support (webcam, IP cameras)
- Label Studio integration for data annotation
- Custom model training pipeline
- Video processing and export
- REST API for inference
- Configuration management

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

### 1. Real-time Detection
```bash
python src/main.py --source camera --device 0
```

### 2. Process Video File
```bash
python src/main.py --source video --input path/to/video.mp4
```

### 3. Start Label Studio
```bash
python src/labeling/start_labelstudio.py
```

### 4. Train Custom Model
```bash
python src/training/train.py --data config/dataset.yaml --epochs 100
```

### 5. Start API Server
```bash
python src/api/app.py
```

## Configuration

Edit `config/config.yaml` to customize:
- Model parameters
- Camera settings
- Tracking parameters
- Label Studio configuration
- Output settings

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