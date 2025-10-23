"""
Label Studio integration for data annotation and labeling
"""
import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
from label_studio_sdk import Client

class LabelStudioManager:
    """Manages Label Studio integration for data annotation"""
    
    def __init__(self, config: Dict):
        self.config = config['label_studio']
        self.url = self.config['url']
        self.api_token = self.config.get('api_token', '')
        self.project_name = self.config['project_name']
        
        # Initialize client
        if self.api_token:
            self.client = Client(url=self.url, api_key=self.api_token)
        else:
            self.client = None
            print("Warning: No API token provided for Label Studio")
        
        self.project = None
        self._setup_project()
    
    def _setup_project(self):
        """Setup or get existing Label Studio project"""
        if not self.client:
            return
        
        try:
            # Get existing project or create new one
            projects = self.client.get_projects()
            
            for project in projects:
                if project.title == self.project_name:
                    self.project = project
                    print(f"Connected to existing project: {self.project_name}")
                    return
            
            # Create new project if not found
            self._create_project()
            
        except Exception as e:
            print(f"Error setting up Label Studio project: {e}")
    
    def _create_project(self):
        """Create new Label Studio project"""
        # Label config for object detection
        label_config = """
        <View>
          <Image name="image" value="$image"/>
          <RectangleLabels name="label" toName="image">
            <Label value="person" background="red"/>
            <Label value="bicycle" background="blue"/>
            <Label value="car" background="green"/>
            <Label value="motorcycle" background="yellow"/>
            <Label value="airplane" background="purple"/>
            <Label value="bus" background="orange"/>
            <Label value="train" background="pink"/>
            <Label value="truck" background="brown"/>
            <Label value="boat" background="cyan"/>
            <Label value="traffic light" background="magenta"/>
            <!-- Add more labels as needed -->
          </RectangleLabels>
        </View>
        """
        
        try:
            self.project = self.client.start_project(
                title=self.project_name,
                label_config=label_config,
                description="YOLOv11 Object Detection Dataset"
            )
            print(f"Created new Label Studio project: {self.project_name}")
        except Exception as e:
            print(f"Error creating Label Studio project: {e}")
    
    def import_detection_results(self, image_paths: List[str], detections_list: List[List]):
        """
        Import detection results as pre-annotations
        
        Args:
            image_paths: List of image file paths
            detections_list: List of detection results for each image
        """
        if not self.project:
            print("No Label Studio project available")
            return
        
        tasks = []
        
        for image_path, detections in zip(image_paths, detections_list):
            # Convert detections to Label Studio format
            predictions = self._convert_detections_to_predictions(detections, image_path)
            
            task = {
                "data": {"image": image_path},
                "predictions": [predictions] if predictions else []
            }
            tasks.append(task)
        
        try:
            # Import tasks to Label Studio
            self.project.import_tasks(tasks)
            print(f"Imported {len(tasks)} tasks to Label Studio")
        except Exception as e:
            print(f"Error importing tasks: {e}")
    
    def _convert_detections_to_predictions(self, detections: List, image_path: str) -> Dict:
        """
        Convert YOLO detections to Label Studio prediction format
        
        Args:
            detections: List of detection objects
            image_path: Path to source image
            
        Returns:
            Label Studio prediction format
        """
        if not detections:
            return {}
        
        # Get image dimensions
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            return {}
        
        height, width = image.shape[:2]
        
        results = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Convert to Label Studio format (percentages)
            x_percent = (x1 / width) * 100
            y_percent = (y1 / height) * 100
            width_percent = ((x2 - x1) / width) * 100
            height_percent = ((y2 - y1) / height) * 100
            
            result = {
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": x_percent,
                    "y": y_percent,
                    "width": width_percent,
                    "height": height_percent,
                    "rectanglelabels": [detection.class_name]
                },
                "score": detection.confidence
            }
            results.append(result)
        
        return {
            "model_version": "yolo11",
            "result": results
        }
    
    def export_annotations(self, export_format: str = "YOLO") -> str:
        """
        Export annotations from Label Studio
        
        Args:
            export_format: Format to export (YOLO, COCO, etc.)
            
        Returns:
            Path to exported annotations
        """
        if not self.project:
            print("No Label Studio project available")
            return ""
        
        try:
            # Export annotations
            export_data = self.project.export_tasks(export_type=export_format)
            
            # Save to file
            export_dir = Path("data/labeled")
            export_dir.mkdir(parents=True, exist_ok=True)
            
            export_file = export_dir / f"annotations_{export_format.lower()}.json"
            
            with open(export_file, 'w') as f:
                if isinstance(export_data, str):
                    f.write(export_data)
                else:
                    json.dump(export_data, f, indent=2)
            
            print(f"Exported annotations to: {export_file}")
            return str(export_file)
            
        except Exception as e:
            print(f"Error exporting annotations: {e}")
            return ""
    
    def convert_to_yolo_format(self, export_file: str, output_dir: str):
        """
        Convert Label Studio export to YOLO format
        
        Args:
            export_file: Path to exported annotations
            output_dir: Directory to save YOLO format files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create directories
        images_dir = output_path / "images"
        labels_dir = output_path / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        try:
            with open(export_file, 'r') as f:
                data = json.load(f)
            
            class_names = set()
            
            for task in data:
                if 'annotations' not in task:
                    continue
                
                image_path = task['data']['image']
                image_name = Path(image_path).stem
                
                # Process annotations
                yolo_annotations = []
                
                for annotation in task['annotations']:
                    for result in annotation.get('result', []):
                        if result['type'] == 'rectanglelabels':
                            value = result['value']
                            class_name = value['rectanglelabels'][0]
                            class_names.add(class_name)
                            
                            # Convert to YOLO format (normalized coordinates)
                            x_center = (value['x'] + value['width'] / 2) / 100
                            y_center = (value['y'] + value['height'] / 2) / 100
                            width_norm = value['width'] / 100
                            height_norm = value['height'] / 100
                            
                            # Get class ID (you may want to define a proper mapping)
                            class_id = list(class_names).index(class_name) if class_name in class_names else 0
                            
                            yolo_annotations.append(f"{class_id} {x_center} {y_center} {width_norm} {height_norm}")
                
                # Save YOLO annotation file
                if yolo_annotations:
                    label_file = labels_dir / f"{image_name}.txt"
                    with open(label_file, 'w') as f:
                        f.write('\n'.join(yolo_annotations))
            
            # Save class names
            classes_file = output_path / "classes.txt"
            with open(classes_file, 'w') as f:
                f.write('\n'.join(sorted(class_names)))
            
            print(f"Converted to YOLO format in: {output_dir}")
            
        except Exception as e:
            print(f"Error converting to YOLO format: {e}")
    
    def get_project_stats(self) -> Dict:
        """Get project statistics"""
        if not self.project:
            return {}
        
        try:
            tasks = self.project.get_tasks()
            
            total_tasks = len(tasks)
            annotated_tasks = len([t for t in tasks if t.get('annotations')])
            
            return {
                'total_tasks': total_tasks,
                'annotated_tasks': annotated_tasks,
                'completion_rate': (annotated_tasks / total_tasks * 100) if total_tasks > 0 else 0
            }
        except Exception as e:
            print(f"Error getting project stats: {e}")
            return {}

def start_label_studio_server():
    """Start Label Studio server"""
    import subprocess
    import sys
    
    try:
        # Start Label Studio server
        subprocess.run([sys.executable, "-m", "label_studio", "start"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting Label Studio server: {e}")
    except FileNotFoundError:
        print("Label Studio not installed. Install with: pip install label-studio")