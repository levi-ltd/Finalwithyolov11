# YOLOv11 Object Detection and Tracking - Development Notes

## Project Structure Overview

This project implements a comprehensive object detection and tracking system using YOLOv11 with Label Studio integration. Here's how the components work together:

### Core Components

1. **Detection Module** (`src/detection/`)
   - `detector.py`: YOLOv11 model wrapper with detection functionality
   - Handles model loading, inference, and result processing
   - Supports different YOLO model sizes (n, s, m, l, x)

2. **Tracking Module** (`src/tracking/`)
   - `tracker.py`: Multi-object tracking using Kalman filters
   - Implements simple centroid tracking with IoU association
   - Maintains object identities across frames

3. **Camera Module** (`src/camera/`)
   - `camera_manager.py`: Camera input and video processing
   - Supports webcam, IP cameras, and video files
   - Handles frame capture, video recording, and streaming

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

### Data Flow

1. **Input**: Camera/video → `CameraManager`
2. **Detection**: Frame → `YOLODetector` → Detection results
3. **Tracking**: Detections → `ObjectTracker` → Tracked objects with IDs
4. **Visualization**: Tracked objects → `Visualizer` → Annotated frame
5. **Output**: Results → File export, API response, or Label Studio

### Key Features

- **Real-time Processing**: Optimized for live camera feeds
- **Multi-object Tracking**: Maintains consistent IDs across frames
- **Label Studio Integration**: Seamless annotation workflow
- **Custom Training**: Fine-tune models on your data
- **REST API**: Easy integration with other systems
- **Comprehensive Logging**: Performance monitoring and debugging
- **Flexible Configuration**: Easy parameter tuning

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