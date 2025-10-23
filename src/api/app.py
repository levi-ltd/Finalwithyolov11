"""
REST API for YOLOv11 detection service
"""
import io
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from detection.detector import YOLODetector
from tracking.tracker import ObjectTracker
from utils.config import load_config
from utils.visualizer import Visualizer

class DetectionAPI:
    """FastAPI application for object detection"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize components
        self.detector = YOLODetector(self.config)
        self.tracker = ObjectTracker(self.config) if self.config['tracking']['enabled'] else None
        self.visualizer = Visualizer(self.config)
        
        # Initialize FastAPI
        self.app = FastAPI(
            title="YOLOv11 Object Detection API",
            description="Real-time object detection and tracking API using YOLOv11",
            version="1.0.0"
        )
        
        # Add CORS middleware
        if self.config['api']['cors_enabled']:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {"message": "YOLOv11 Detection API", "status": "running"}
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "model_loaded": self.detector.model is not None}
        
        @self.app.post("/detect")
        async def detect_objects(file: UploadFile = File(...)):
            """Detect objects in uploaded image"""
            try:
                # Read image
                image_data = await file.read()
                image = self._decode_image(image_data)
                
                # Run detection
                detections = self.detector.detect(image)
                
                # Format results
                results = self._format_detections(detections)
                
                return JSONResponse(content={
                    "status": "success",
                    "detections": results,
                    "count": len(detections)
                })
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/detect_and_visualize")
        async def detect_and_visualize(file: UploadFile = File(...)):
            """Detect objects and return annotated image"""
            try:
                # Read image
                image_data = await file.read()
                image = self._decode_image(image_data)
                
                # Run detection
                detections = self.detector.detect(image)
                
                # Annotate image
                annotated_image = self.visualizer.draw_detections(image, detections)
                
                # Encode image
                encoded_image = self._encode_image(annotated_image)
                
                return StreamingResponse(
                    io.BytesIO(encoded_image),
                    media_type="image/jpeg",
                    headers={"Content-Disposition": "attachment; filename=detection_result.jpg"}
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/track")
        async def track_objects(file: UploadFile = File(...)):
            """Track objects in uploaded image (requires previous context)"""
            if not self.tracker:
                raise HTTPException(status_code=400, detail="Tracking not enabled")
            
            try:
                # Read image
                image_data = await file.read()
                image = self._decode_image(image_data)
                
                # Run detection
                detections = self.detector.detect(image)
                
                # Run tracking
                tracked_objects = self.tracker.update(detections)
                
                # Format results
                results = self._format_tracked_objects(tracked_objects)
                
                return JSONResponse(content={
                    "status": "success",
                    "tracked_objects": results,
                    "count": len(tracked_objects)
                })
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/config")
        async def get_config():
            """Get current configuration"""
            return JSONResponse(content=self.config)
        
        @self.app.post("/config/update")
        async def update_config(new_config: Dict):
            """Update configuration"""
            try:
                # Update specific config values
                for key, value in new_config.items():
                    if key in self.config:
                        self.config[key].update(value)
                
                return JSONResponse(content={
                    "status": "success",
                    "message": "Configuration updated"
                })
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models")
        async def get_available_models():
            """Get list of available models"""
            models_dir = Path("models")
            models = []
            
            if models_dir.exists():
                for model_file in models_dir.rglob("*.pt"):
                    models.append({
                        "name": model_file.stem,
                        "path": str(model_file),
                        "size": model_file.stat().st_size
                    })
            
            return JSONResponse(content={"models": models})
    
    def _decode_image(self, image_data: bytes) -> np.ndarray:
        """Decode image data to OpenCV format"""
        try:
            # Try PIL first
            image = Image.open(io.BytesIO(image_data))
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return image_cv
        except Exception:
            # Try OpenCV
            nparr = np.frombuffer(image_data, np.uint8)
            image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image_cv is None:
                raise ValueError("Could not decode image")
            return image_cv
    
    def _encode_image(self, image: np.ndarray) -> bytes:
        """Encode OpenCV image to bytes"""
        _, buffer = cv2.imencode('.jpg', image)
        return buffer.tobytes()
    
    def _format_detections(self, detections: List) -> List[Dict]:
        """Format detection results for API response"""
        results = []
        
        for detection in detections:
            result = {
                "class_id": detection.class_id,
                "class_name": detection.class_name,
                "confidence": float(detection.confidence),
                "bbox": {
                    "x1": float(detection.bbox[0]),
                    "y1": float(detection.bbox[1]),
                    "x2": float(detection.bbox[2]),
                    "y2": float(detection.bbox[3])
                }
            }
            results.append(result)
        
        return results
    
    def _format_tracked_objects(self, tracked_objects: List) -> List[Dict]:
        """Format tracking results for API response"""
        results = []
        
        for obj in tracked_objects:
            result = {
                "track_id": obj.track_id,
                "class_id": obj.class_id,
                "class_name": obj.class_name,
                "confidence": float(obj.confidence),
                "bbox": {
                    "x1": float(obj.bbox[0]),
                    "y1": float(obj.bbox[1]),
                    "x2": float(obj.bbox[2]),
                    "y2": float(obj.bbox[3])
                }
            }
            results.append(result)
        
        return results

def create_app(config_path: str = "config/config.yaml") -> FastAPI:
    """Create FastAPI application"""
    api = DetectionAPI(config_path)
    return api.app

if __name__ == "__main__":
    import uvicorn
    
    # Load config
    config = load_config("config/config.yaml")
    api_config = config['api']
    
    # Create app
    app = create_app()
    
    # Run server
    uvicorn.run(
        app,
        host=api_config['host'],
        port=api_config['port'],
        debug=api_config['debug']
    )