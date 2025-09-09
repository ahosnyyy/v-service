import os
import yaml
from typing import List, Dict, Tuple, Any, Optional

# Add detector directory to Python path for local imports
import sys
from pathlib import Path
detector_dir = Path(__file__).parent.absolute()
if str(detector_dir) not in sys.path:
    sys.path.insert(0, str(detector_dir))

# Import configuration
from config import config

def load_clo_values(yaml_file: str = None) -> Dict[str, Any]:
    """
    Load CLO values from a YAML file.
    
    Args:
        yaml_file: Path to the YAML file containing CLO values
        
    Returns:
        Dictionary containing CLO values and base CLO
    """
    if yaml_file is None:
        yaml_file = config.clo.values_file
    
    if not os.path.exists(yaml_file):
        # Try relative to detector directory
        detector_dir = os.path.dirname(os.path.abspath(__file__))
        alt_yaml_file = os.path.join(detector_dir, "clo_values.yaml")
        if os.path.exists(alt_yaml_file):
            yaml_file = alt_yaml_file
            print(f"Using CLO values from detector directory: {yaml_file}")
        else:
            raise FileNotFoundError(f"CLO values file not found: {yaml_file} or {alt_yaml_file}")
    
    with open(yaml_file, 'r') as f:
        clo_data = yaml.safe_load(f)
    
    return clo_data

def get_clothing_clo_values(yaml_file: str = None) -> Dict[str, float]:
    """
    Returns a dictionary mapping clothing items to their CLO values.
    
    Args:
        yaml_file: Path to the YAML file containing CLO values
        
    Returns:
        Dictionary mapping clothing items to their CLO values
    """
    clo_data = load_clo_values(yaml_file)
    return clo_data.get('clo_values', {})

def get_base_clo_value(yaml_file: str = None) -> float:
    """
    Returns the base CLO value for a nude person.
    
    Args:
        yaml_file: Path to the YAML file containing CLO values
        
    Returns:
        Base CLO value
    """
    clo_data = load_clo_values(yaml_file)
    return clo_data.get('base_clo', 0.0)

def map_detections_to_clo(detections: List[Any], yaml_file: str = None) -> Tuple[List[Any], float]:
    """
    Maps detected clothing items to their CLO values and calculates total CLO.
    If no lower body item is detected, adds a default pants CLO value.
    
    Args:
        detections: List of DetectionResult objects
        yaml_file: Path to the YAML file containing CLO values
        
    Returns:
        Tuple of (updated_detections, total_clo_value)
    """
    clo_mapping = get_clothing_clo_values(yaml_file)
    base_clo = get_base_clo_value(yaml_file)
    total_clo = 0.0
    
    # Define lower body items
    lower_body_items = ['pants', 'shorts', 'skirt']
    
    # Track if we've found a lower body item
    found_lower_body = False
    
    for detection in detections:
        # Convert class name to lowercase for case-insensitive matching
        class_name_lower = detection.class_name.lower()
        
        # Check if this is a lower body item
        if class_name_lower in lower_body_items:
            found_lower_body = True
        
        # Assign CLO value if class is in our mapping
        if class_name_lower in clo_mapping:
            detection.clo_value = clo_mapping[class_name_lower]
            total_clo += detection.clo_value
        else:
            detection.clo_value = 0.0
    
    # If no lower body item was detected, add default pants CLO value
    if not found_lower_body and 'pants' in clo_mapping:
        # Add pants CLO value to total
        pants_clo = clo_mapping['pants']
        total_clo += pants_clo
        
        # Add a note about this in the console for debugging
        print(f"No lower body item detected, adding default pants CLO value: {pants_clo}")
    
    # Add base CLO value
    total_clo += base_clo
    
    return detections, total_clo

