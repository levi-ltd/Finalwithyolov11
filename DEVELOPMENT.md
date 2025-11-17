# YOLOv11 Singer Detection System - Development Notes

## Project Overview

This project implements a specialized object detection and tracking system using YOLOv11, specifically designed for **singer detection** in live performance environments. The system uses a simplified 3-class approach that automatically identifies singers based on person-microphone proximity.

### Core Concept: Automatic Singer Detection

**Input Classes**: Person, Microphone  
**Logic**: Person + Microphone (within proximity threshold) = Singer  
**Output Classes**: Person, Microphone, Singer  

### Key Components

1. **Singer Detection Module** (`src/detection/detector.py`)
   - Processes YOLO detections for person and microphone objects
   - Implements proximity-based singer classification algorithm
   - Maps original YOLO classes to simplified 3-class system
   - Configurable distance thresholds for singer detection

2. **Enhanced Tracking** (`src/tracking/tracker.py`)
   - Maintains singer state across frames
   - Tracks microphone-person relationships over time
   - Preserves singer IDs even when microphone is temporarily occluded
   - Stores singer-specific metadata (microphone distance, original class)

3. **Performance-Optimized Camera** (`src/camera/camera_manager.py`)
   - Optimized for live performance scenarios
   - Supports stage lighting conditions
   - Real-time processing for live streaming integration

4. **Label Studio Integration** (`src/labeling/`)
   - `label_studio_manager.py`: Interface with Label Studio API
   - `start_labelstudio.py`: Helper to start Label Studio server
   - Handles data import/export and annotation management

5. **Training Module** (`src/training/`)
   - `train.py`: Custom model training utilities
   - Dataset preparation and validation
   - Model export and conversion

6. **API Module** (`src/api/`)
   - `app.py`: FastAPI REST service for detection/tracking
   - Supports image upload, real-time processing
   - RESTful endpoints for integration

7. **Utilities** (`src/utils/`)
   - `config.py`: Configuration management
   - `logger.py`: Logging and performance tracking
   - `visualizer.py`: Result visualization and annotation

### Configuration

All settings are managed through `config/config.yaml`:
- Model parameters (thresholds, device selection)
- Camera settings (resolution, FPS)
- Tracking parameters (max age, IoU thresholds)
- Output configuration (video saving, annotations)
- Label Studio connection details
- API server settings

### Singer Detection Data Flow

1. **Input**: Camera/video → `CameraManager`
2. **YOLO Detection**: Frame → `YOLODetector` → Raw detections (person, microphone)
3. **Singer Classification**: Raw detections → Proximity analysis → Singer detection
4. **Tracking**: Enhanced detections → `ObjectTracker` → Tracked singers with IDs
5. **Visualization**: Tracked objects → `Visualizer` → Color-coded annotations
6. **Output**: Results → Performance analytics, streaming overlay, or Label Studio

### Singer Detection Algorithm

```python
def detect_singers(detections):
    persons = [d for d in detections if d.class_name == 'person']
    micros = [d for d in detections if d.class_name == 'micro']
    
    for person in persons:
        for micro in micros:
            distance = calculate_distance(person.bbox, micro.bbox)
            if distance < PROXIMITY_THRESHOLD:
                # Convert person to singer
                person.class_name = 'singer'
                person.has_micro = True
                person.micro_distance = distance
                break
```

### Key Features for Live Performances

- **Real-time Processing**: Optimized for live streaming (target: 30+ FPS)
- **Singer ID Persistence**: Maintains consistent IDs as performers move
- **Stage-aware Detection**: Handles varying lighting and angles
- **Multi-singer Support**: Tracks multiple singers simultaneously
- **Microphone Handoff Detection**: Detects when microphone changes hands

### Performance Considerations

- **Model Size**: Balance between speed and accuracy
  - YOLOv11n: Fastest, lowest accuracy (~100 FPS)
  - YOLOv11s: Good balance (~80 FPS)  
  - YOLOv11m: Better accuracy (~60 FPS)
  - YOLOv11l/x: Best accuracy (~30-40 FPS)

- **Image Size**: Smaller input images = faster processing
- **Tracking Parameters**: Tune based on object movement speed
- **Device**: GPU acceleration significantly improves performance

### Extension Points

1. **Custom Tracking Algorithms**
   - Replace simple tracker with DeepSORT, ByteTrack, etc.
   - Implement in `src/tracking/`

2. **Additional Input Sources**
   - RTSP streams, image sequences
   - Extend `CameraManager` class

3. **Advanced Visualization**
   - Heatmaps, trajectory plots, statistics overlays
   - Add to `Visualizer` class

4. **Model Optimization**
   - TensorRT, ONNX export for faster inference
   - Quantization and pruning

5. **Cloud Integration**
   - AWS/Azure computer vision services
   - Cloud storage for results

### Troubleshooting

**Common Issues:**
- Camera not detected: Check device permissions and index
- Low FPS: Reduce image size or use smaller model
- Tracking ID switches: Tune IoU threshold and max age
- Label Studio connection: Verify API token and URL

**Performance Optimization:**
- Use GPU if available
- Adjust confidence thresholds
- Limit max detections per frame
- Use threading for camera capture

**Memory Management:**
- Clear old tracking data periodically
- Limit video buffer size
- Use image compression for storage

### Development Workflow

1. **Setup**: Run `python setup.py` for initial configuration
2. **Testing**: Use notebooks for experimentation
3. **Development**: Modify modules in `src/`
4. **Configuration**: Update `config/config.yaml` as needed
5. **Deployment**: Use API or standalone script

### Integration Examples

**With Label Studio:**
```python
from src.labeling.label_studio_manager import LabelStudioManager

ls_manager = LabelStudioManager(url, token)
ls_manager.import_detection_results(images, detections)
```

**With Custom Training:**
```python
from src.training.train import YOLOTrainer

trainer = YOLOTrainer(config)
model_path = trainer.train(dataset_config)
```

**API Integration:**
```bash
# Start API server
python src/api/app.py

# Use API
curl -X POST "http://localhost:8000/detect" -F "file=@image.jpg"
```

This architecture provides a solid foundation for building production-ready object detection and tracking systems while maintaining flexibility for customization and extension.