"""
Multi-object tracking using ByteTrack algorithm
"""
import numpy as np
from typing import List, Dict
from collections import OrderedDict
from filterpy.kalman import KalmanFilter

class TrackState:
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3

class Track:
    """Single object track with singer support"""
    
    def __init__(self, detection, track_id: int):
        self.track_id = track_id
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        
        # Singer-specific attributes
        self.has_micro = getattr(detection, 'has_micro', False)
        self.micro_distance = getattr(detection, 'micro_distance', 0.0)
        self.original_class = getattr(detection, 'original_class', None)
        
        # Track state
        self.state = TrackState.NEW
        self.age = 0
        self.hits = 0
        self.time_since_update = 0
        
        # Initialize Kalman filter
        self.kf = self._init_kalman_filter(detection.bbox)
        
    def _init_kalman_filter(self, bbox):
        """Initialize Kalman filter for tracking"""
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (constant velocity model)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Measurement noise
        kf.R[2:, 2:] *= 10.
        
        # Process noise
        kf.P[4:, 4:] *= 1000.
        kf.P *= 10.
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01
        
        # Initialize state
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        kf.x[:4] = [cx, cy, w, h]
        
        return kf
    
    def predict(self):
        """Predict next state"""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        
    def update(self, detection):
        """Update track with new detection"""
        x1, y1, x2, y2 = detection.bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        self.kf.update([cx, cy, w, h])
        
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.class_name = detection.class_name
        self.class_id = detection.class_id
        
        # Update singer-specific attributes
        self.has_micro = getattr(detection, 'has_micro', False)
        self.micro_distance = getattr(detection, 'micro_distance', 0.0)
        self.original_class = getattr(detection, 'original_class', None)
        
        self.hits += 1
        self.time_since_update = 0
        
        if self.state == TrackState.NEW and self.hits >= 3:
            self.state = TrackState.TRACKED
    
    def get_state(self):
        """Get current bounding box state"""
        cx, cy, w, h = self.kf.x[:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2])
    
    def mark_missed(self):
        """Mark track as missed"""
        if self.state == TrackState.NEW:
            self.state = TrackState.REMOVED
        elif self.state == TrackState.TRACKED:
            self.state = TrackState.LOST

class ObjectTracker:
    """Multi-object tracker using ByteTrack algorithm"""
    
    def __init__(self, config: Dict):
        self.config = config['tracking']
        self.max_age = self.config['max_age']
        self.min_hits = self.config['min_hits']
        self.iou_threshold = self.config['iou_threshold']
        
        self.tracks = []
        self.track_id_counter = 0
        
    def update(self, detections: List) -> List:
        """
        Update tracker with new detections
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List of tracked objects with track IDs
        """
        # Predict all existing tracks
        for track in self.tracks:
            track.predict()
        
        # Associate detections with tracks
        matched_pairs, unmatched_dets, unmatched_trks = self._associate_detections_to_tracks(
            detections, self.tracks
        )
        
        # Update matched tracks
        for det_idx, trk_idx in matched_pairs:
            self.tracks[trk_idx].update(detections[det_idx])
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_new_track(detections[det_idx])
        
        # Mark unmatched tracks as missed
        for trk_idx in unmatched_trks:
            self.tracks[trk_idx].mark_missed()
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks if not self._should_remove_track(track)]
        
        # Return active tracks
        active_tracks = []
        for track in self.tracks:
            if track.state == TrackState.TRACKED:
                # Create detection-like object with track ID
                tracked_detection = type('TrackedDetection', (), {})()
                tracked_detection.bbox = track.get_state()
                tracked_detection.confidence = track.confidence
                tracked_detection.class_id = track.class_id
                tracked_detection.class_name = track.class_name
                tracked_detection.track_id = track.track_id
                
                # Add singer-specific attributes
                tracked_detection.has_micro = track.has_micro
                tracked_detection.micro_distance = track.micro_distance
                tracked_detection.original_class = track.original_class
                
                active_tracks.append(tracked_detection)
        
        return active_tracks
    
    def _associate_detections_to_tracks(self, detections, tracks):
        """Associate detections to tracks using IoU"""
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(tracks)))
        for d, det in enumerate(detections):
            for t, track in enumerate(tracks):
                iou_matrix[d, t] = self._calculate_iou(det.bbox, track.get_state())
        
        # Hungarian algorithm or greedy matching
        matched_pairs = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(tracks)))
        
        # Simple greedy matching
        for _ in range(min(len(detections), len(tracks))):
            if iou_matrix.size == 0:
                break
            
            # Find best match
            max_iou_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[max_iou_idx]
            
            if max_iou >= self.iou_threshold:
                det_idx, trk_idx = max_iou_idx
                matched_pairs.append((det_idx, trk_idx))
                unmatched_dets.remove(det_idx)
                unmatched_trks.remove(trk_idx)
                
                # Set matched rows and columns to 0
                iou_matrix[det_idx, :] = 0
                iou_matrix[:, trk_idx] = 0
            else:
                break
        
        return matched_pairs, unmatched_dets, unmatched_trks
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU)"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _create_new_track(self, detection):
        """Create new track from detection"""
        track = Track(detection, self.track_id_counter)
        self.track_id_counter += 1
        self.tracks.append(track)
    
    def _should_remove_track(self, track):
        """Determine if track should be removed"""
        if track.state == TrackState.REMOVED:
            return True
        if track.state == TrackState.LOST and track.time_since_update > self.max_age:
            return True
        return False