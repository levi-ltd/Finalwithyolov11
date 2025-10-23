"""
Simple script to start Label Studio server
"""
import sys
import subprocess
from pathlib import Path

def start_labelstudio():
    """Start Label Studio server"""
    try:
        # Create Label Studio project directory
        project_dir = Path("data/labelstudio")
        project_dir.mkdir(parents=True, exist_ok=True)
        
        print("Starting Label Studio server...")
        print("Access Label Studio at: http://localhost:8080")
        print("Press Ctrl+C to stop the server")
        
        # Start Label Studio
        cmd = [sys.executable, "-m", "label_studio", "start", "--port", "8080"]
        subprocess.run(cmd, cwd=project_dir, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error starting Label Studio: {e}")
        print("Make sure Label Studio is installed: pip install label-studio")
    except KeyboardInterrupt:
        print("\nLabel Studio server stopped")
    except FileNotFoundError:
        print("Label Studio not found. Install with: pip install label-studio")

if __name__ == "__main__":
    start_labelstudio()