"""
Visualization utilities for drawing detection results
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
import random

class Visualizer:
    """Handles visualization of detection and tracking results"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Class-specific colors for our 3-class system
        self.class_colors = {
            'person': (0, 0, 255),    # Red
            'micro': (255, 0, 0),     # Blue
            'singer': (0, 255, 0)     # Green
        }
        
        # Fallback color palette for track IDs
        self.colors = self._generate_colors(100)
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        
    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for different classes"""
        colors = []
        for i in range(num_colors):
            # Generate colors in HSV space for better distribution
            hue = int(180 * i / num_colors)
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color_bgr)))
        return colors
    
    def draw_detections(self, frame: np.ndarray, detections: List) -> np.ndarray:
        """
        Draw detection boxes and labels on frame with singer detection support
        
        Args:
            frame: Input frame
            detections: List of detection objects
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, detection.bbox)
            
            # Get color based on class name first, then fallback to class ID
            color = self.class_colors.get(detection.class_name, self.colors[detection.class_id % len(self.colors)])
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, self.thickness)
            
            # Prepare label text with singer information
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            if hasattr(detection, 'track_id') and detection.track_id is not None:
                label = f"ID:{detection.track_id} {label}"
            
            # Add singer-specific information
            if detection.class_name == 'singer' and hasattr(detection, 'has_micro') and detection.has_micro:
                micro_dist = getattr(detection, 'micro_distance', 0)
                label += f" (🎤 {micro_dist:.0f}px)"
            
            # Calculate label size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.thickness
            )
            
            # Draw label background
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - baseline - 2),
                self.font,
                self.font_scale,
                (255, 255, 255),
                self.thickness
            )
            
            # Draw center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(annotated_frame, (center_x, center_y), 4, color, -1)
            
            # Special indicator for singers
            if detection.class_name == 'singer':
                # Draw microphone icon indicator (simplified)
                cv2.putText(annotated_frame, "♪", (x2 - 20, y1 + 15), 
                           self.font, 0.8, (0, 255, 0), 2)
        
        return annotated_frame
    
    def draw_tracks(self, frame: np.ndarray, tracks: List) -> np.ndarray:
        """
        Draw tracking trails for objects
        
        Args:
            frame: Input frame
            tracks: List of track objects with history
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for track in tracks:
            if hasattr(track, 'history') and len(track.history) > 1:
                # Draw track trail
                points = np.array(track.history, dtype=np.int32)
                color = self.colors[track.class_id % len(self.colors)]
                
                # Draw polyline for track
                cv2.polylines(annotated_frame, [points], False, color, 2)
                
                # Draw circles at track points
                for point in points[-10:]:  # Last 10 points
                    cv2.circle(annotated_frame, tuple(point), 3, color, -1)
        
        return annotated_frame
    
    def draw_info_panel(self, frame: np.ndarray, info: Dict) -> np.ndarray:
        """
        Draw information panel on frame
        
        Args:
            frame: Input frame
            info: Dictionary containing information to display
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Panel dimensions
        panel_height = 100
        panel_width = 300
        
        # Draw panel background
        panel_overlay = annotated_frame.copy()
        cv2.rectangle(
            panel_overlay,
            (10, 10),
            (10 + panel_width, 10 + panel_height),
            (0, 0, 0),
            -1
        )
        
        # Blend panel with frame
        cv2.addWeighted(panel_overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
        
        # Draw panel border
        cv2.rectangle(
            annotated_frame,
            (10, 10),
            (10 + panel_width, 10 + panel_height),
            (255, 255, 255),
            2
        )
        
        # Draw information text
        y_offset = 35
        line_height = 20
        
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(
                annotated_frame,
                text,
                (20, y_offset),
                self.font,
                0.5,
                (255, 255, 255),
                1
            )
            y_offset += line_height
        
        return annotated_frame
    
    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """
        Draw FPS counter on frame
        
        Args:
            frame: Input frame
            fps: Current FPS value
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        fps_text = f"FPS: {fps:.1f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            fps_text, self.font, 0.7, 2
        )
        
        # Position at top right
        h, w = frame.shape[:2]
        x = w - text_width - 20
        y = text_height + 20
        
        # Draw background
        cv2.rectangle(
            annotated_frame,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + baseline + 5),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            annotated_frame,
            fps_text,
            (x, y),
            self.font,
            0.7,
            (0, 255, 0),
            2
        )
        
        return annotated_frame
    
    def save_annotated_frame(self, frame: np.ndarray, output_path: str):
        """Save annotated frame to file"""
        cv2.imwrite(output_path, frame)