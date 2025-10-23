"""
Model training utilities for YOLOv11
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Optional
from ultralytics import YOLO

class YOLOTrainer:
    """YOLOv11 model trainer"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.training_config = config['training']
        self.model_config = config['model']
        
        # Training parameters
        self.data_path = self.training_config['data_path']
        self.epochs = self.training_config['epochs']
        self.batch_size = self.training_config['batch_size']
        self.image_size = self.training_config['image_size']
        self.learning_rate = self.training_config['learning_rate']
        self.save_period = self.training_config['save_period']
        
        # Initialize model
        self.model = None
        
    def prepare_dataset(self, dataset_dir: str) -> str:
        """
        Prepare dataset configuration file
        
        Args:
            dataset_dir: Directory containing the dataset
            
        Returns:
            Path to dataset configuration file
        """
        dataset_path = Path(dataset_dir)
        
        # Create dataset.yaml file
        dataset_config = {
            'path': str(dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 80,  # Number of classes (COCO default)
            'names': [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
        }
        
        # Check if custom classes file exists
        classes_file = dataset_path / "classes.txt"
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                custom_classes = [line.strip() for line in f.readlines()]
            dataset_config['names'] = custom_classes
            dataset_config['nc'] = len(custom_classes)
        
        # Save dataset configuration
        config_file = dataset_path / "dataset.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        return str(config_file)
    
    def train(self, data_config: str, pretrained_weights: Optional[str] = None) -> str:
        """
        Train YOLOv11 model
        
        Args:
            data_config: Path to dataset configuration file
            pretrained_weights: Path to pretrained weights (optional)
            
        Returns:
            Path to trained model
        """
        # Initialize model
        if pretrained_weights:
            self.model = YOLO(pretrained_weights)
        else:
            # Use pretrained model from config
            self.model = YOLO(self.model_config['weights'])
        
        print(f"Starting training with {self.epochs} epochs...")
        print(f"Dataset: {data_config}")
        print(f"Batch size: {self.batch_size}")
        print(f"Image size: {self.image_size}")
        
        # Create output directory
        output_dir = Path("models/trained")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Train the model
            results = self.model.train(
                data=data_config,
                epochs=self.epochs,
                batch=self.batch_size,
                imgsz=self.image_size,
                lr0=self.learning_rate,
                save_period=self.save_period,
                project=str(output_dir),
                name="yolo11_custom",
                exist_ok=True,
                verbose=True
            )
            
            # Get path to best model
            best_model_path = results.save_dir / "weights" / "best.pt"
            print(f"Training completed! Best model saved at: {best_model_path}")
            
            return str(best_model_path)
            
        except Exception as e:
            print(f"Error during training: {e}")
            return ""
    
    def validate(self, model_path: str, data_config: str) -> Dict:
        """
        Validate trained model
        
        Args:
            model_path: Path to trained model
            data_config: Path to dataset configuration file
            
        Returns:
            Validation metrics
        """
        try:
            # Load model
            model = YOLO(model_path)
            
            # Run validation
            results = model.val(
                data=data_config,
                imgsz=self.image_size,
                verbose=True
            )
            
            # Extract metrics
            metrics = {
                'mAP50': results.box.map50,
                'mAP50-95': results.box.map,
                'precision': results.box.mp,
                'recall': results.box.mr,
                'f1': results.box.f1
            }
            
            print("Validation Results:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"Error during validation: {e}")
            return {}
    
    def resume_training(self, checkpoint_path: str) -> str:
        """
        Resume training from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Path to resumed model
        """
        try:
            # Resume training
            self.model = YOLO(checkpoint_path)
            results = self.model.train(resume=True)
            
            best_model_path = results.save_dir / "weights" / "best.pt"
            print(f"Resume training completed! Best model saved at: {best_model_path}")
            
            return str(best_model_path)
            
        except Exception as e:
            print(f"Error resuming training: {e}")
            return ""
    
    def export_model(self, model_path: str, format: str = "onnx") -> str:
        """
        Export model to different formats
        
        Args:
            model_path: Path to trained model
            format: Export format (onnx, tensorrt, etc.)
            
        Returns:
            Path to exported model
        """
        try:
            model = YOLO(model_path)
            
            # Export model
            exported_path = model.export(format=format)
            print(f"Model exported to: {exported_path}")
            
            return exported_path
            
        except Exception as e:
            print(f"Error exporting model: {e}")
            return ""

def split_dataset(dataset_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        dataset_dir: Directory containing images and labels
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set (rest goes to test)
    """
    import random
    import shutil
    
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    if not images_dir.exists() or not labels_dir.exists():
        print("Images or labels directory not found")
        return
    
    # Get all image files
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    random.shuffle(image_files)
    
    total_files = len(image_files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    
    # Create split directories
    for split in ['train', 'val', 'test']:
        (dataset_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_path / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Split files
    for i, image_file in enumerate(image_files):
        label_file = labels_dir / f"{image_file.stem}.txt"
        
        if i < train_count:
            split = "train"
        elif i < train_count + val_count:
            split = "val"
        else:
            split = "test"
        
        # Move image file
        shutil.move(str(image_file), str(dataset_path / "images" / split / image_file.name))
        
        # Move label file if exists
        if label_file.exists():
            shutil.move(str(label_file), str(dataset_path / "labels" / split / label_file.name))
    
    print(f"Dataset split completed:")
    print(f"  Train: {train_count} images")
    print(f"  Val: {val_count} images")
    print(f"  Test: {total_files - train_count - val_count} images")