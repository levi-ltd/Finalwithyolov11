"""
Quick setup script for YOLOv11 Object Detection and Tracking
Run this script to initialize the project and check dependencies
"""
import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is supported"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        # Install main requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/labeled", 
        "data/processed",
        "data/yolo_dataset/images/train",
        "data/yolo_dataset/images/val",
        "data/yolo_dataset/images/test",
        "data/yolo_dataset/labels/train",
        "data/yolo_dataset/labels/val",
        "data/yolo_dataset/labels/test",
        "models/trained",
        "outputs",
        "results",
        "logs"
    ]
    
    print("📁 Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")

def download_sample_model():
    """Download YOLOv11 model weights"""
    print("🔄 Downloading YOLO 11 model weights...")
    
    try:
        from ultralytics import YOLO
        
        # Download nano model (smallest, fastest)
        model = YOLO("yolo11n.pt")
        print("✅ YOLOv11n model downloaded")
        
        # Test the model
        print("🧪 Testing model...")
        import numpy as np
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(test_image, verbose=False)
        print("✅ Model test successful")
        
        return True
    except Exception as e:
        print(f"❌ Failed to download/test model: {e}")
        return False

def check_camera():
    """Check if camera is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ Camera available")
                return True
        print("⚠️ Camera not available (will use simulation mode)")
        return False
    except Exception as e:
        print(f"⚠️ Camera check failed: {e}")
        return False

def create_demo_script():
    """Create a simple demo script"""
    demo_script = """#!/usr/bin/env python3
\"\"\"
Simple demo script for YOLOv11 detection
\"\"\"
import sys
sys.path.append('src')

from src.main import main

if __name__ == "__main__":
    # Run with default camera
    main()
"""
    
    with open("demo.py", "w") as f:
        f.write(demo_script)
    
    print("✅ Demo script created: demo.py")

def main():
    """Main setup function"""
    print("🚀 YOLOv11 Object Detection & Tracking Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Download model
    if not download_sample_model():
        return False
    
    # Check camera
    check_camera()
    
    # Create demo script
    create_demo_script()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📝 Quick Start:")
    print("   1. Run the demo: python demo.py")
    print("   2. Open the Jupyter notebook: jupyter notebook notebooks/yolo11_detection_tracking_tutorial.ipynb")
    print("   3. Start Label Studio: python src/labeling/start_labelstudio.py")
    print("   4. Start API server: python src/api/app.py")
    
    print("\n📚 Documentation:")
    print("   - README.md for detailed instructions")
    print("   - config/config.yaml for configuration")
    print("   - notebooks/ for tutorials and examples")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)