"""
Logging utilities
"""
import logging
import sys
from pathlib import Path
from typing import Dict

def setup_logger(config: Dict, name: str = "yolo_tracker") -> logging.Logger:
    """
    Setup logger with configuration
    
    Args:
        config: Configuration dictionary
        name: Logger name
        
    Returns:
        Configured logger
    """
    logging_config = config.get('logging', {})
    level = logging_config.get('level', 'INFO')
    save_logs = logging_config.get('save_logs', False)
    log_file = logging_config.get('log_file', 'logs/app.log')
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if enabled
    if save_logs:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics = {}
    
    def log_fps(self, fps: float):
        """Log FPS metric"""
        self.metrics['fps'] = fps
        self.logger.info(f"FPS: {fps:.2f}")
    
    def log_detection_time(self, time_ms: float):
        """Log detection time"""
        self.metrics['detection_time'] = time_ms
        self.logger.debug(f"Detection time: {time_ms:.2f}ms")
    
    def log_tracking_time(self, time_ms: float):
        """Log tracking time"""
        self.metrics['tracking_time'] = time_ms
        self.logger.debug(f"Tracking time: {time_ms:.2f}ms")
    
    def log_frame_processing_time(self, time_ms: float):
        """Log total frame processing time"""
        self.metrics['frame_time'] = time_ms
        self.logger.debug(f"Frame processing time: {time_ms:.2f}ms")
    
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        return self.metrics.copy()