#!/usr/bin/env python3
"""
Detector Client

Client class for communicating with the FastAPI detection service.
"""

import requests
import json
import time
import os
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
from PIL import Image
import io
import base64

from shared.utils import setup_logging, load_config


class DetectorClient:
    """Client for communicating with the FastAPI detection service."""
    
    def __init__(self, config_path: str = "detector/config.yaml"):
        """Initialize the detector client."""
        self.config = load_config(config_path)
        
        # Get logging configuration
        log_level = self.config.get("logging", {}).get("level", "WARNING")
        log_file = self.config.get("logging", {}).get("file")
        self.logger = setup_logging("detector_client", log_level, log_file)
        
        # API configuration
        # Always connect to localhost, regardless of server binding
        self.api_host = "localhost"  # Always use localhost for client
        self.api_port = self.config.get("api", {}).get("port", 8000)
        self.api_base_url = f"http://{self.api_host}:{self.api_port}"
        
        # Model configuration
        self.model_config = self.config.get("model", {})
        self.onnx_file = self.model_config.get("onnx_file", "model.onnx")
        self.img_size = self.model_config.get("img_size", 640)
        self.conf_thres = self.model_config.get("conf_thres", 0.6)
        self.device = self.model_config.get("device", "cpu")
        self.categories_file = self.model_config.get("categories_file", "categories.yaml")
        
        # Service status
        self.running = False
        self.last_health_check = 0
        self.health_check_interval = 30  # seconds
        self.frame_count = 0  # Counter for processed frames
        
        self.logger.info(f"Detector client initialized for {self.api_base_url}")
    
    def start(self) -> bool:
        """Start the detector client (check if service is available)."""
        try:
            self.logger.info("Starting detector client...")
            
            # Check if the detection service is available
            if self._check_service_health():
                self.running = True
                self.logger.info("Detector client started successfully")
                return True
            else:
                self.logger.error("Detection service is not available")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start detector client: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the detector client."""
        self.logger.info("Stopping detector client...")
        self.running = False
        self.logger.info("Detector client stopped")
    
    def is_running(self) -> bool:
        """Check if the detector client is running."""
        return self.running
    
    def is_ready(self) -> bool:
        """Check if the detector service is ready to process requests."""
        if not self.running:
            return False
        
        # Check health periodically
        current_time = time.time()
        if current_time - self.last_health_check > self.health_check_interval:
            self.last_health_check = current_time
            return self._check_service_health()
        
        return True
    
    def _check_service_health(self) -> bool:
        """Check if the detection service is healthy."""
        try:
            response = requests.get(f"{self.api_base_url}/", timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.debug(f"Health check failed: {e}")
            return False
    
    def detect_frame(self, frame_data, frame_name: str = None) -> Optional[Dict[str, Any]]:
        """
        Send a frame to the detection service for processing.
        
        Args:
            frame_data: Frame data (PIL Image, numpy array, or bytes)
            frame_name: Optional name for the frame (for JSON naming)
            
        Returns:
            Detection results or None if failed
        """
        try:
            if not self.is_ready():
                self.logger.warning("Detector service not ready")
                return None
            
            # Convert frame data to bytes if needed
            if hasattr(frame_data, 'save'):  # PIL Image
                img_bytes = io.BytesIO()
                frame_data.save(img_bytes, format='JPEG', quality=85)
                img_bytes = img_bytes.getvalue()
            elif hasattr(frame_data, 'tobytes'):  # numpy array
                # Convert numpy array to PIL Image first
                if len(frame_data.shape) == 3:
                    img = Image.fromarray(frame_data)
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='JPEG', quality=85)
                    img_bytes = img_bytes.getvalue()
                else:
                    self.logger.error("Unsupported numpy array shape")
                    return None
            elif isinstance(frame_data, bytes):
                img_bytes = frame_data
            else:
                self.logger.error(f"Unsupported frame data type: {type(frame_data)}")
                return None
            
            # Prepare the request
            filename = frame_name if frame_name else 'frame.jpg'
            files = {'file': (filename, img_bytes, 'image/jpeg')}
            
            # Include frame_name in the form data as well to ensure it's passed correctly
            data = {
                'onnx_file': self.onnx_file,
                'img_size': self.img_size,
                'conf_thres': self.conf_thres,
                'device': self.device,
                'categories_file': self.categories_file
            }
            
            # Add frame_name to form data if provided
            if frame_name:
                data['frame_name'] = frame_name
            
            # Make the API request
            response = requests.post(
                f"{self.api_base_url}/detect",
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                detection_count = len(result.get('detections', []))
                # Increment frame count
                self.frame_count += 1
                # Only log every 10th frame to reduce log volume
                if self.frame_count % 10 == 0:
                    self.logger.info(f"{detection_count} objects in frame {self.frame_count}")
                return result
            else:
                self.logger.error(f"Detection failed: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.error("Detection request timed out")
            return None
        except requests.exceptions.ConnectionError:
            self.logger.error("Connection error to detection service")
            self.running = False  # Mark as not running
            return None
        except Exception as e:
            self.logger.error(f"Error during detection: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the detector client and service."""
        return {
            "running": self.running,
            "service_ready": self.is_ready(),
            "api_url": self.api_base_url,
            "last_health_check": self.last_health_check,
            "model_config": {
                "onnx_file": self.onnx_file,
                "img_size": self.img_size,
                "conf_thres": self.conf_thres,
                "device": self.device
            }
        }


# For backward compatibility, you can also create a simple wrapper
class Detector:
    """Wrapper class for backward compatibility with existing coordinator code."""
    
    def __init__(self, config_path: str = "detector/config.yaml"):
        self.client = DetectorClient(config_path)
    
    def start(self) -> bool:
        return self.client.start()
    
    def stop(self) -> None:
        self.client.stop()
    
    def is_running(self) -> bool:
        return self.client.is_running()
    
    def is_ready(self) -> bool:
        return self.client.is_ready()
    
    def detect_frame(self, frame_data, frame_name: str = None) -> Optional[Dict[str, Any]]:
        return self.client.detect_frame(frame_data, frame_name)
    
    def get_status(self) -> Dict[str, Any]:
        return self.client.get_status()
