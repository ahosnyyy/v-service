import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    """Model configuration settings"""
    onnx_file: str = Field(default="model.onnx", description="Path to ONNX model file")
    img_size: int = Field(default=640, ge=128, le=1024, description="Input image size")
    conf_thres: float = Field(default=0.6, ge=0.0, le=1.0, description="Confidence threshold")
    device: str = Field(default="cpu", description="Device to use for inference (cpu/gpu)")

class PathsConfig(BaseModel):
    """File path configuration settings"""
    categories_file: str = Field(default="categories.yaml", description="Path to categories file")
    default_image: str = Field(default="./default/default_img.jpg", description="Path to default image")
    output_directory: str = Field(default="output", description="Output directory path")

class InferenceConfig(BaseModel):
    """Inference settings"""
    batch_size: int = Field(default=1, ge=1, le=32, description="Batch size for inference")
    max_detections: int = Field(default=100, ge=1, le=1000, description="Maximum number of detections")

class APIConfig(BaseModel):
    """API server configuration"""
    host: str = Field(default="0.0.0.0", description="API host address")
    port: int = Field(default=8000, ge=1024, le=65535, description="API port number")
    reload: bool = Field(default=True, description="Enable auto-reload for development")

class CLOConfig(BaseModel):
    """CLO processing configuration"""
    enabled: bool = Field(default=True, description="Enable CLO value processing")
    values_file: str = Field(default="clo_values.yaml", description="Path to CLO values file")

class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field(default="INFO", description="Log level")
    enable_console: bool = Field(default=True, description="Enable console logging")
    enable_file: bool = Field(default=False, description="Enable file logging")
    file: str = Field(default="./logs/detector.log", description="Log file path")

class SaveResultsConfig(BaseModel):
    """Detection results saving configuration"""
    enabled: bool = Field(default=True, description="Enable saving detection results to JSON files")
    output_dir: str = Field(default="detections", description="Directory to save detection results")

class DetectorConfig(BaseModel):
    """Main detector configuration"""
    model: ModelConfig = Field(default_factory=ModelConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    clo: CLOConfig = Field(default_factory=CLOConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    save_results: SaveResultsConfig = Field(default_factory=SaveResultsConfig)

    @classmethod
    def from_yaml(cls, yaml_file: str = "config.yaml") -> "DetectorConfig":
        """Load configuration from YAML file"""
        try:
            import yaml
            # Get detector directory for relative path resolution
            detector_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(detector_dir, yaml_file)
            
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            return cls(**data)
        except Exception as e:
            print(f"Error loading config from {yaml_file}: {e}")
            print("Using default configuration")
            return cls()



    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.model_dump()

# Global configuration instance
config = DetectorConfig.from_yaml()



# Export for easy access
__all__ = ['DetectorConfig', 'config']
