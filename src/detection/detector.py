"""
YOLOv11 Object Detector
"""
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
import cv2

class Detection:
    """Class to represent a detection with singer-specific attributes"""
    def __init__(self, bbox, confidence, class_id, class_name):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.track_id = None  # Will be set by tracker
        
        # Singer-specific attributes
        self.has_micro = False
        self.micro_distance = 0.0
        self.original_class = None

class YOLODetector:
    """YOLOv11 Object Detector with Singer Detection"""
    
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
        
        # Simplified class mapping for person, micro, and singer
        self.class_mapping = {'person': 0, 'micro': 1, 'singer': 2}
        self.proximity_threshold = config.get('singer_detection', {}).get('proximity_threshold', 50)
        
        # Original YOLO class names for filtering
        self.yolo_class_names = self.model.names
        
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run object detection on a frame with singer detection logic
        
        Args:
            frame: Input image as numpy array
            
        Returns:
            List of Detection objects including detected singers
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
        
        raw_detections = []
        
        # Process YOLO results and filter for our classes
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box data
                    xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Get original YOLO class name
                    yolo_class_name = self.yolo_class_names.get(class_id, f"class_{class_id}")
                    
                    # Map to our simplified classes
                    if yolo_class_name == 'person':
                        mapped_class_name = 'person'
                        mapped_class_id = 0
                    elif yolo_class_name in ['microphone', 'mic', 'cell phone']:  # Include phones as mics
                        mapped_class_name = 'micro'
                        mapped_class_id = 1
                    else:
                        continue  # Skip other classes
                    
                    # Create detection object
                    detection = Detection(
                        bbox=xyxy,
                        confidence=conf,
                        class_id=mapped_class_id,
                        class_name=mapped_class_name
                    )
                    
                    raw_detections.append(detection)
        
        # Apply singer detection logic
        final_detections = self._detect_singers(raw_detections)
        
        return final_detections
    
    def _detect_singers(self, detections: List[Detection]) -> List[Detection]:
        """
        Detect singers based on person-microphone proximity
        
        Args:
            detections: List of raw detections
            
        Returns:
            List of detections with singers identified
        """
        final_detections = []
        person_detections = []
        micro_detections = []
        
        # Separate persons and microphones
        for detection in detections:
            if detection.class_name == 'person':
                person_detections.append(detection)
            elif detection.class_name == 'micro':
                micro_detections.append(detection)
        
        used_micros = set()
        
        # Check each person for nearby microphone
        for person in person_detections:
            closest_micro = None
            min_distance = float('inf')
            closest_micro_idx = None
            
            # Find closest microphone
            for i, micro in enumerate(micro_detections):
                if i in used_micros:
                    continue
                    
                distance = self._calculate_distance(person.bbox, micro.bbox)
                
                if distance < min_distance and distance < self.proximity_threshold:
                    min_distance = distance
                    closest_micro = micro
                    closest_micro_idx = i
            
            if closest_micro is not None:
                # Convert person with microphone to singer
                singer_detection = Detection(
                    bbox=person.bbox,
                    confidence=person.confidence,
                    class_id=2,  # Singer class ID
                    class_name='singer'
                )
                singer_detection.has_micro = True
                singer_detection.micro_distance = min_distance
                singer_detection.original_class = 'person'
                
                final_detections.append(singer_detection)
                used_micros.add(closest_micro_idx)
                
                # Still add the microphone as separate detection
                final_detections.append(closest_micro)
            else:
                # Add person without microphone
                final_detections.append(person)
        
        # Add remaining unused microphones
        for i, micro in enumerate(micro_detections):
            if i not in used_micros:
                final_detections.append(micro)
        
        return final_detections
    
    def _calculate_distance(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Calculate distance between centers of two bounding boxes
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            Distance between centers
        """
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2
        
        return ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
    
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