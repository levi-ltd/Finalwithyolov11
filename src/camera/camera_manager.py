"""
Camera input management for real-time detection and video processing
"""
import cv2
import numpy as np
from typing import Optional, Dict
import threading
import queue
import time

class CameraManager:
    """Manages camera input and video processing"""
    
    def __init__(self, config: Dict, device: Optional[int] = None, video_path: Optional[str] = None):
        self.config = config['camera']
        self.output_config = config['output']
        
        self.device = device
        self.video_path = video_path
        self.cap = None
        self.out = None
        
        # Threading for better performance
        self.frame_queue = queue.Queue(maxsize=self.config['buffer_size'])
        self.capture_thread = None
        self.running = False
        
        self._initialize_source()
        self._initialize_output()
        
    def _initialize_source(self):
        """Initialize video source"""
        if self.video_path:
            # Video file input
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video file: {self.video_path}")
        else:
            # Camera input
            self.cap = cv2.VideoCapture(self.device if self.device is not None else self.config['source'])
            if not self.cap.isOpened():
                raise ValueError(f"Could not open camera device: {self.device}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['height'])
            self.cap.set(cv2.CAP_PROP_FPS, self.config['fps'])
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config['buffer_size'])
        
        # Get actual properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Initialized source: {self.width}x{self.height} @ {self.fps} FPS")
        
    def _initialize_output(self):
        """Initialize video output if enabled"""
        if self.output_config['save_video']:
            output_dir = self.output_config['output_dir']
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"detection_output_{timestamp}.mp4")
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*self.output_config['video_codec'])
            self.out = cv2.VideoWriter(
                output_path,
                fourcc,
                self.output_config['video_fps'],
                (self.width, self.height)
            )
            
            print(f"Video output will be saved to: {output_path}")
    
    def start_capture_thread(self):
        """Start threaded frame capture for better performance"""
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
    
    def _capture_frames(self):
        """Capture frames in separate thread"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Add frame to queue (non-blocking)
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                # Remove oldest frame if queue is full
                try:
                    self.frame_queue.get(block=False)
                    self.frame_queue.put(frame, block=False)
                except queue.Empty:
                    pass
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get next frame from source
        
        Returns:
            Frame as numpy array or None if no frame available
        """
        if self.capture_thread and self.capture_thread.is_alive():
            # Get frame from thread queue
            try:
                frame = self.frame_queue.get(timeout=1.0)
                return frame
            except queue.Empty:
                return None
        else:
            # Direct capture
            ret, frame = self.cap.read()
            return frame if ret else None
    
    def write_frame(self, frame: np.ndarray):
        """Write frame to output video"""
        if self.out is not None:
            self.out.write(frame)
    
    def get_frame_info(self) -> Dict:
        """Get information about the video source"""
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.video_path else None,
            'current_frame': int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) if self.video_path else None
        }
    
    def set_frame_position(self, frame_number: int):
        """Set current frame position (for video files only)"""
        if self.video_path:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    def release(self):
        """Release resources"""
        self.running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        if self.out:
            self.out.release()
        
        print("Camera resources released")

class IPCameraManager(CameraManager):
    """Extended camera manager for IP cameras"""
    
    def __init__(self, config: Dict, ip_url: str, username: str = "", password: str = ""):
        self.ip_url = ip_url
        self.username = username
        self.password = password
        
        # Construct full URL with authentication
        if username and password:
            # Format: rtsp://username:password@ip:port/path
            auth_url = ip_url.replace("://", f"://{username}:{password}@")
            super().__init__(config, video_path=auth_url)
        else:
            super().__init__(config, video_path=ip_url)
    
    def reconnect(self):
        """Reconnect to IP camera"""
        if self.cap:
            self.cap.release()
        
        self._initialize_source()
        print("Reconnected to IP camera")