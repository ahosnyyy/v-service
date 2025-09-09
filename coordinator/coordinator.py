#!/usr/bin/env python3
"""
Vision Service Coordinator

Orchestrates the recorder and detector services, manages frame processing,
and handles the overall lifecycle of the vision service.
"""

import threading
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from shared.buffer import FrameBuffer
from shared.utils import setup_logging, load_config
from recorder.recorder import Recorder
from detector.detector_client import Detector
from detector.detector import set_frame_buffer


class VisionCoordinator:
    """Main coordinator for the vision service."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the coordinator with configuration."""
        self.config = load_config(config_path)
        
        # Get logging configuration
        log_level = self.config.get("logging", {}).get("level", "INFO")
        log_file = self.config.get("logging", {}).get("file")
        self.logger = setup_logging("coordinator", log_level, log_file)
        
        # Service components
        self.recorder: Optional[Recorder] = None
        self.detector: Optional[Detector] = None
        
        # Shared resources
        self.frame_buffer: Optional[FrameBuffer] = None
        
        # Control flags
        self.running = False
        self.processing_thread: Optional[threading.Thread] = None
        
        # Configuration
        # First try to get from detector.fps (new), then fall back to detection_interval (deprecated)
        detector_config = self.config.get("detector", {})
        self.detection_fps = detector_config.get("fps", self.config.get("detection_interval", 1))  # Detection FPS
        self.buffer_size = self.config.get("buffer_size", 100)
        self.processing_delay = self.config.get("processing_delay", 0.1)  # seconds
        
        # Calculate detection interval based on FPS
        self.detection_interval = 1.0 / self.detection_fps if self.detection_fps > 0 else 1.0
        self.last_detection_time = 0
        
        # Store latest detection results for UI
        self.last_detection_result = None
        self.last_detection_frame = None  # Store the frame that was sent to detection
        self.detection_stats = {
            'total_detections': 0,
            'last_detection_time': None,
            'detection_counts': {}
        }
        
        self.logger.debug("Coordinator initialized")
    
    def start(self) -> bool:
        """Start the vision service."""
        try:
            # Initialize shared buffer with longer timeout
            self.frame_buffer = FrameBuffer(maxlen=self.buffer_size, timeout=5.0)
            
            # Start recorder service (uses its own config file)
            self.recorder = Recorder(
                frame_buffer=self.frame_buffer,
                detection_fps=self.detection_fps  # Pass detector FPS to recorder
            )
            self.recorder.start()
            
            # Start detector service (uses its own config file)
            self.detector = Detector()
            self.detector.start()
            
            # Pass frame buffer reference to detector service
            set_frame_buffer(self.frame_buffer)
            
            # Disable automatic frame processing - only manual detection via /detect
            self.running = True
            # Start frame buffer population thread (sends frames to detector's buffer)
            self.buffer_thread = threading.Thread(
                target=self._populate_detector_buffer,
                daemon=True,
                name="DetectorBufferPopulator"
            )
            self.buffer_thread.start()
            self.logger.info("Vision Service ready (manual detection only)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Vision Service: {e}")
            self.stop()
            return False
    
    def stop(self) -> None:
        """Stop the vision service."""
        self.running = False
        self.logger.info("Stopping Vision Service")
        
        # Stop frame processing
        if hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
            self.logger.info("Frame processing stopped")
        
        # Stop detector buffer population
        if hasattr(self, 'buffer_thread') and self.buffer_thread and self.buffer_thread.is_alive():
            self.buffer_thread.join(timeout=5.0)
            self.logger.info("Detector buffer population stopped")
        
        # Stop detector service
        if self.detector:
            self.detector.stop()
            self.logger.info("Detector service stopped")
        
        # Stop recorder service
        if self.recorder:
            self.recorder.stop()
            self.logger.info("Recorder service stopped")
        
        # Clear buffer
        if self.frame_buffer:
            self.frame_buffer.clear()
        
        self.logger.info("Vision Service stopped")
    
    # def _add_frame_to_detector_buffer(self, frame, frame_name: str):
    #     """Add frame to detector's buffer via HTTP API."""
    #     try:
    #         import requests
    #         import io
    #         from PIL import Image
    #         
    #         # Convert frame to bytes
    #         if hasattr(frame, 'save'):  # PIL Image
    #             img_bytes = io.BytesIO()
    #             frame.save(img_bytes, format='JPEG', quality=85)
    #             frame_bytes = img_bytes.getvalue()
    #         elif hasattr(frame, 'tobytes'):  # numpy array
    #             if len(frame.shape) == 3:
    #                 img = Image.fromarray(frame)
    #                 img_bytes = io.BytesIO()
    #                 img.save(img_bytes, format='JPEG', quality=85)
    #                 frame_bytes = img_bytes.getvalue()
    #             else:
    #                 self.logger.error("Unsupported numpy array shape for detector buffer")
    #                 return
    #         elif isinstance(frame, bytes):
    #             frame_bytes = frame
    #         else:
    #             self.logger.error(f"Unsupported frame data type for detector buffer: {type(frame)}")
    #             return
    #         
    #         # Send frame to detector's buffer
    #         detector_url = f"http://localhost:{self.detector.client.api_port}/add-frame"
    #         response = requests.post(
    #             detector_url,
    #             data={"frame_name": frame_name},
    #             files={"file": (frame_name, frame_bytes, "image/jpeg")},
    #             timeout=5
    #         )
    #         
    #         if response.status_code == 200:
    #             self.logger.debug(f"Added frame {frame_name} to detector buffer")
    #         else:
    #             self.logger.warning(f"Failed to add frame to detector buffer: {response.status_code}")
    #             try:
    #                 error_detail = response.json()
    #                 self.logger.warning(f"Error detail: {error_detail}")
    #             except:
    #                 self.logger.warning(f"Error response: {response.text}")
    #                 
    #     except Exception as e:
    #         self.logger.error(f"Error adding frame to detector buffer: {e}")
    
    def _populate_detector_buffer(self) -> None:
        """Populate detector's buffer with frames from coordinator's buffer."""
        frame_count = 0
        last_buffer_log = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                if not self.frame_buffer or self.frame_buffer.empty():
                    # Log buffer status every 5 seconds
                    if current_time - last_buffer_log >= 5.0:
                        self.logger.info("Coordinator buffer is empty. Waiting for frames...")
                        last_buffer_log = current_time
                    time.sleep(0.1)
                    continue
                
                # Get frame from coordinator's buffer
                frame_data = self.frame_buffer.get_latest()
                
                if frame_data is None:
                    continue
                
                frame_count += 1
                
                # Extract frame data and name
                if isinstance(frame_data, tuple) and len(frame_data) == 2:
                    frame, frame_name = frame_data
                else:
                    frame = frame_data
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    frame_name = f"frame_{timestamp}.jpg"
                
                # Send frame to detector's buffer (disabled - endpoint hidden)
                # if self.detector and self.detector.is_ready():
                #     self._add_frame_to_detector_buffer(frame, frame_name)
                
                # Log progress every 30 frames
                if frame_count % 30 == 0:
                    self.logger.info(f"Populated detector buffer with {frame_count} frames")
                
                # Small delay to avoid overwhelming the detector
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in detector buffer population: {e}")
                time.sleep(0.1)
    
    def _process_frames(self) -> None:
        """Main frame processing loop."""
        frame_count = 0
        last_buffer_log = 0
        last_buffer_check = 0
        buffer_check_interval = 1.0  # Check buffer utilization every second
        high_utilization_threshold = 0.8  # 80% buffer capacity
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check buffer utilization periodically and take action if needed
                if current_time - last_buffer_check >= buffer_check_interval:
                    if self.frame_buffer and not self.frame_buffer.empty():
                        buffer_stats = self.frame_buffer.get_stats()
                        utilization = buffer_stats.get('utilization', 0)
                        
                        # If buffer is getting full, clear older frames more aggressively
                        if utilization > high_utilization_threshold:
                            self.logger.warning(f"Buffer utilization high ({utilization:.2f}), clearing older frames")
                            # Keep only the most recent frames (e.g., last 10%)
                            frames_to_keep = max(1, int(self.frame_buffer.maxlen * 0.1))
                            frames_to_skip = len(self.frame_buffer) - frames_to_keep
                            
                            if frames_to_skip > 0:
                                # Keep only the newest frames by removing older ones
                                for _ in range(frames_to_skip):
                                    if not self.frame_buffer.empty():
                                        self.frame_buffer.get()  # Remove oldest frame
                                    else:
                                        break
                                self.logger.info(f"Cleared {frames_to_skip} older frames, keeping {frames_to_keep} newest frames")
                        
                        # Log buffer statistics periodically
                        self.logger.info(f"Buffer stats: size={len(self.frame_buffer)}/{self.frame_buffer.maxlen}, "
                                        f"utilization={utilization:.2f}, pushed={buffer_stats.get('total_pushed', 0)}, "
                                        f"popped={buffer_stats.get('total_popped', 0)}, "
                                        f"dropped={buffer_stats.get('dropped_frames', 0)}")
                    
                    last_buffer_check = current_time
                
                if not self.frame_buffer or self.frame_buffer.empty():
                    # Log buffer status every 5 seconds
                    if current_time - last_buffer_log >= 5.0:
                        self.logger.info(f"Buffer is empty. Waiting for frames...")
                        last_buffer_log = current_time
                    time.sleep(self.processing_delay)
                    continue
                
                # Get frame from buffer - ALWAYS use get_latest to process the freshest frame
                buffer_stats = self.frame_buffer.get_stats() if self.frame_buffer else {}
                utilization = buffer_stats.get('utilization', 0)
                
                # Always use get_latest to process the most recent frame
                frame_data = self.frame_buffer.get_latest()
                
                # Log buffer pressure status periodically
                if utilization > high_utilization_threshold:
                    if frame_count % 10 == 0:  # Log less frequently when under pressure
                        self.logger.warning(f"Buffer pressure: {utilization:.2f}, processing latest frame")
                elif frame_count % 30 == 0:  # Log during normal operation
                    self.logger.info(f"Buffer utilization: {utilization:.2f}, processing latest frame")
                
                if frame_data is None:
                    continue
                
                frame_count += 1
                
                # Extract frame data and name
                if isinstance(frame_data, tuple) and len(frame_data) == 2:
                    frame, frame_name = frame_data
                    self.logger.debug(f"Frame name from buffer: {frame_name}")
                else:
                    frame = frame_data
                    # Generate a more unique frame name with timestamp
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    frame_name = f"frame_{timestamp}.jpg"  # More unique fallback name
                    self.logger.debug(f"Generated frame name: {frame_name}")
                
                # Log frame received less frequently to reduce log volume
                if frame_count % 30 == 0:
                    buffer_stats = self.frame_buffer.get_stats() if self.frame_buffer else {}
                    self.logger.info(f"Received frame {frame_count} ({frame_name}) from buffer. Buffer size: {len(self.frame_buffer)}")
                    self.logger.info(f"Buffer stats: {buffer_stats}")
                
                # Process frame
                self.logger.info(f"Processing frame {frame_count} ({frame_name}) at {self.detection_fps} FPS")
                
                # Send frame to detector's buffer via HTTP API
                if self.detector and self.detector.is_ready():
                    try:
                        # Also add frame to detector's buffer (disabled - endpoint hidden)
                        # self._add_frame_to_detector_buffer(frame, frame_name)
                        
                        # Ensure frame_name is passed correctly
                        self.logger.debug(f"Sending frame {frame_name} to detector")
                        detection_result = self.detector.detect_frame(frame, frame_name)
                        
                        if detection_result:
                            # Store latest detection result and frame for UI
                            self.last_detection_result = detection_result
                            self.last_detection_frame = frame  # Store the frame that was sent to detection
                            self.detection_stats['last_detection_time'] = current_time
                            
                            num_detections = len(detection_result.get('detections', []))
                            self.detection_stats['total_detections'] += num_detections
                            
                            # Update detection counts by class
                            for det in detection_result.get('detections', []):
                                # Try different possible keys for class name
                                class_name = det.get('class') or det.get('class_name') or det.get('label') or det.get('name') or 'unknown'
                                if class_name not in self.detection_stats['detection_counts']:
                                    self.detection_stats['detection_counts'][class_name] = 0
                                self.detection_stats['detection_counts'][class_name] += 1
                            
                            self.logger.info(f"Detection completed for frame {frame_count} ({frame_name}): {num_detections} objects detected")
                            
                            # Log more details about detection results only when objects are detected
                            if num_detections > 0:
                                self.logger.info(f"Detection details: {detection_result}")
                        else:
                            self.logger.warning(f"No detection result returned for frame {frame_name}")
                    except Exception as e:
                        self.logger.error(f"Detection failed for frame {frame_count} ({frame_name}): {e}")
                        import traceback
                        self.logger.error(f"Traceback: {traceback.format_exc()}")
                else:
                    self.logger.warning("Detector not ready, skipping frame")
                
                # Update last detection time
                self.last_detection_time = current_time
                
                # Adaptive delay based on buffer utilization
                if utilization < 0.3:  # Buffer is relatively empty
                    # Process faster to catch up
                    time.sleep(max(0.001, self.processing_delay * 0.5))
                else:
                    # Normal processing delay
                    time.sleep(self.processing_delay)
                
            except Exception as e:
                self.logger.error(f"Error in frame processing loop: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                time.sleep(self.processing_delay)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the vision service."""
        status = {
            "running": self.running,
            "recorder_active": self.recorder.is_running() if self.recorder else False,
            "detector_active": self.detector.is_running() if self.detector else False,
            "buffer_size": len(self.frame_buffer) if self.frame_buffer else 0,
            "buffer_capacity": self.buffer_size,
            "detection_fps": self.detection_fps,
            "detection_interval_seconds": self.detection_interval
        }
        return status
    
    def is_running(self) -> bool:
        """Check if the service is running."""
        return self.running


def main():
    """Main entry point for the vision service."""
    coordinator = VisionCoordinator()
    
    try:
        if coordinator.start():
            # Keep the service running
            while coordinator.is_running():
                time.sleep(1)
        else:
            print("Failed to start Vision Service")
            return 1
            
    except KeyboardInterrupt:
        print("\nShutting down Vision Service...")
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    finally:
        coordinator.stop()
    
    return 0


if __name__ == "__main__":
    exit(main())
