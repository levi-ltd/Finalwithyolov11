"""
Configuration management utilities
"""
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid, raises exception otherwise
    """
    required_sections = ['model', 'camera', 'tracking', 'classes', 'output']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate model section
    model_config = config['model']
    required_model_keys = ['name', 'weights', 'conf_threshold', 'iou_threshold']
    
    for key in required_model_keys:
        if key not in model_config:
            raise ValueError(f"Missing required model configuration: {key}")
    
    # Validate thresholds
    if not 0.0 <= model_config['conf_threshold'] <= 1.0:
        raise ValueError("conf_threshold must be between 0.0 and 1.0")
    
    if not 0.0 <= model_config['iou_threshold'] <= 1.0:
        raise ValueError("iou_threshold must be between 0.0 and 1.0")
    
    return True

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override with
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged