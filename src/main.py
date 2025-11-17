"""
Main entry point for YOLOv11 Singer Detection and Tracking System
Detects persons, microphones, and automatically identifies singers
"""
import argparse
import cv2
import yaml
from pathlib import Path

from detection.detector import YOLODetector
from tracking.tracker import ObjectTracker
from camera.camera_manager import CameraManager
from utils.logger import setup_logger
from utils.config import load_config
from utils.visualizer import Visualizer

def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Object Detection and Tracking")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--source", type=str, choices=["camera", "video", "image"], default="camera", help="Input source type")
    parser.add_argument("--input", type=str, help="Path to input file (for video/image source)")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--output", type=str, help="Output path for results")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger = setup_logger(config)
    
    logger.info("Starting YOLOv11 Detection and Tracking System")
    
    # Initialize components
    detector = YOLODetector(config)
    tracker = ObjectTracker(config) if config['tracking']['enabled'] else None
    visualizer = Visualizer(config)
    
    # Initialize camera manager
    if args.source == "camera":
        camera_manager = CameraManager(config, device=args.device)
    elif args.source == "video":
        if not args.input:
            raise ValueError("Input video path required for video source")
        camera_manager = CameraManager(config, video_path=args.input)
    else:
        raise NotImplementedError("Image source not implemented yet")
    
    # Main processing loop
    frame_count = 0
    try:
        while True:
            frame = camera_manager.get_frame()
            if frame is None:
                break
            
            frame_count += 1
            
            # Run detection
            detections = detector.detect(frame)
            
            # Run tracking if enabled
            if tracker and len(detections) > 0:
                tracked_objects = tracker.update(detections)
            else:
                tracked_objects = detections
            
            # Count class-specific objects
            class_counts = {'person': 0, 'micro': 0, 'singer': 0}
            for obj in tracked_objects:
                if obj.class_name in class_counts:
                    class_counts[obj.class_name] += 1
            
            # Visualize results
            annotated_frame = visualizer.draw_detections(frame, tracked_objects)
            
            # Add class count info
            info_panel = {
                'Frame': frame_count,
                'Total Objects': len(tracked_objects),
                'Persons': class_counts['person'],
                'Microphones': class_counts['micro'],
                'Singers': class_counts['singer']
            }
            annotated_frame = visualizer.draw_info_panel(annotated_frame, info_panel)
            
            # Display frame
            cv2.imshow("YOLOv11 Singer Detection System", annotated_frame)
            
            # Save frame if configured
            if config['output']['save_video']:
                camera_manager.write_frame(annotated_frame)
            
            # Print periodic statistics
            if frame_count % 30 == 0:
                logger.info(f"Frame {frame_count}: {class_counts['singer']} singers, {class_counts['person']} persons, {class_counts['micro']} mics")
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        camera_manager.release()
        cv2.destroyAllWindows()
        logger.info("System shutdown complete")

if __name__ == "__main__":
    main()