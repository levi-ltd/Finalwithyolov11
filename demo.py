#!/usr/bin/env python3
"""
Simple demo script for YOLOv11 Object Detection and Tracking
This script provides a quick way to test the system with different input sources.
"""
import sys
import argparse
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def run_camera_demo():
    """Run detection on camera feed"""
    print("🎥 Starting camera detection demo...")
    
    try:
        from main import main
        
        # Simulate command line arguments for camera input
        sys.argv = ["demo.py", "--source", "camera", "--device", "0"]
        main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure to install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Demo error: {e}")

def run_video_demo(video_path):
    """Run detection on video file"""
    print(f"🎬 Starting video detection demo: {video_path}")
    
    try:
        from main import main
        
        # Simulate command line arguments for video input
        sys.argv = ["demo.py", "--source", "video", "--input", video_path]
        main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure to install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Demo error: {e}")

def run_api_demo():
    """Start the API server"""
    print("🌐 Starting API server demo...")
    
    try:
        import subprocess
        import sys
        
        # Start the API server
        subprocess.run([sys.executable, "src/api/app.py"])
        
    except Exception as e:
        print(f"❌ API demo error: {e}")

def run_label_studio_demo():
    """Start Label Studio server"""
    print("🏷️ Starting Label Studio demo...")
    
    try:
        import subprocess
        import sys
        
        # Start Label Studio
        subprocess.run([sys.executable, "src/labeling/start_labelstudio.py"])
        
    except Exception as e:
        print(f"❌ Label Studio demo error: {e}")

def check_setup():
    """Check if the project is properly set up"""
    print("🔍 Checking project setup...")
    
    # Check if source directory exists
    if not Path("src").exists():
        print("❌ Source directory not found. Run setup.py first.")
        return False
    
    # Check if config exists
    if not Path("config/config.yaml").exists():
        print("❌ Configuration file not found.")
        return False
    
    # Try to import main modules
    try:
        sys.path.insert(0, "src")
        from detection.detector import YOLODetector
        from tracking.tracker import ObjectTracker
        print("✅ Core modules available")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install dependencies: pip install -r requirements.txt")
        return False
    
    print("✅ Project setup looks good!")
    return True

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(
        description="YOLOv11 Object Detection and Tracking Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py camera          # Use default camera
  python demo.py video path.mp4  # Process video file
  python demo.py api             # Start API server
  python demo.py labelstudio     # Start Label Studio
  python demo.py check           # Check setup
        """
    )
    
    parser.add_argument(
        "mode", 
        choices=["camera", "video", "api", "labelstudio", "check"],
        help="Demo mode to run"
    )
    
    parser.add_argument(
        "input",
        nargs="?",
        help="Input file path (for video mode)"
    )
    
    args = parser.parse_args()
    
    print("🚀 YOLOv11 Object Detection & Tracking Demo")
    print("=" * 50)
    
    if args.mode == "check":
        check_setup()
        
    elif args.mode == "camera":
        if check_setup():
            run_camera_demo()
            
    elif args.mode == "video":
        if not args.input:
            print("❌ Video file path required for video mode")
            print("💡 Usage: python demo.py video path/to/video.mp4")
            return
        
        if not Path(args.input).exists():
            print(f"❌ Video file not found: {args.input}")
            return
            
        if check_setup():
            run_video_demo(args.input)
            
    elif args.mode == "api":
        if check_setup():
            run_api_demo()
            
    elif args.mode == "labelstudio":
        run_label_studio_demo()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Try running 'python demo.py check' to verify setup")