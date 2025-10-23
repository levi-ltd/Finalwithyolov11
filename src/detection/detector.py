"""
YOLOv11 Object Detector
"""
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
import cv2

class Detection:
    """Class to represent a detection"""
    def __init__(self, bbox, confidence, class_id, class_name):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.track_id = None  # Will be set by tracker

class YOLODetector:
    """YOLOv11 Object Detector"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_config = config['model']
        
        # Load YOLO model
        self.model = YOLO(self.model_config['weights'])
        
        # Set device
        if self.model_config['device'] == 'auto':
            self.device = 'cuda' if self.model.device.type == 'cuda' else 'cpu'
        else:
            self.device = self.model_config['device']
            
        # Detection parameters
        self.conf_threshold = self.model_config['conf_threshold']
        self.iou_threshold = self.model_config['iou_threshold']
        self.max_detections = self.model_config['max_detections']
        
        # Enabled classes
        self.enabled_classes = config['classes']['enabled_classes']
        self.class_names = config['classes']['class_names']
        
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run object detection on a frame
        
        Args:
            frame: Input image as numpy array
            
        Returns:
            List of Detection objects
        """
        # Run inference
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box data
                    xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Filter by enabled classes
                    if self.enabled_classes and class_id not in self.enabled_classes:
                        continue
                    
                    # Get class name
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    
                    # Create detection object
                    detection = Detection(
                        bbox=xyxy,
                        confidence=conf,
                        class_id=class_id,
                        class_name=class_name
                    )
                    
                    detections.append(detection)
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Run detection on multiple frames
        
        Args:
            frames: List of input images
            
        Returns:
            List of detection lists for each frame
        """
        all_detections = []
        
        for frame in frames:
            detections = self.detect(frame)
            all_detections.append(detections)
            
        return all_detections