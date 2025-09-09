import cv2
import yaml
import os
import time
import threading
from datetime import datetime
from pathlib import Path
import logging
import sys

# Add project root to path for shared utilities
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from shared.utils import setup_logging

# Global config path
GLOBAL_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

class Recorder:
    def __init__(self, frame_buffer=None, detection_fps=10, config_path="recorder/config.yaml"):
        """Initialize the recorder with configuration from YAML file."""
        # Store frame buffer first (before any logging setup)
        self.frame_buffer = frame_buffer
        self.detection_fps = detection_fps
        
        # Load config first to get logging settings
        self.config = self._load_config(config_path)
        
        # Setup logging using shared utilities with config settings
        log_level = self.config.get("logging", {}).get("level", "INFO")
        log_file = self.config.get("logging", {}).get("file")
        self.logger = setup_logging("recorder", log_level, log_file)
        
        # Now that logger is initialized, log that config was loaded
        self.logger.info(f"Configuration loaded from {config_path}")
        if hasattr(self, '_config_load_error'):
            self.logger.warning(self._config_load_error)
            delattr(self, '_config_load_error')
        self.cap = None
        self.output_folder = None
        self.frame_count = 0
        self.start_time = None
        self.running = False
        self.capture_thread = None
        
        # Frame rate control for buffer
        self.last_buffer_time = 0
        # Calculate interval between frames that should be added to buffer
        self.buffer_interval = 1.0 / self.detection_fps if self.detection_fps > 0 else 1.0
        self.logger.info(f"Buffer interval set to {self.buffer_interval:.3f}s ({self.detection_fps} FPS)")
        # Track frames added to buffer vs total frames captured
        self.buffer_frames_added = 0
        
        # Log frame buffer status
        if self.frame_buffer is not None:
            self.logger.info(f"Frame buffer provided to recorder: {type(self.frame_buffer).__name__}")
        else:
            self.logger.warning("No frame buffer provided to recorder")
        
        # Create output directory
        self._setup_output_directory()
        
    def _load_config(self, config_path):
        """Load configuration from YAML file and merge with global config."""
        # First load recorder-specific config
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            # Store error for later logging
            self._config_load_error = f"Config file {config_path} not found. Using default configuration."
            print(self._config_load_error)
            config = self._get_default_config()
        except yaml.YAMLError as e:
            # Store error for later logging
            self._config_load_error = f"Error parsing config file: {e}. Using default configuration."
            print(self._config_load_error)
            config = self._get_default_config()
            
        # Now load global config for FPS and buffer_size
        try:
            with open(GLOBAL_CONFIG_PATH, 'r') as file:
                global_config = yaml.safe_load(file)
                
                # Override recorder FPS with global setting if available
                if 'recorder' in global_config and 'fps' in global_config['recorder']:
                    if 'recorder' not in config:
                        config['recorder'] = {}
                    config['recorder']['fps'] = global_config['recorder']['fps']
                    
                # Use global buffer_size if available
                if 'buffer_size' in global_config:
                    config['recorder']['buffer_size'] = global_config['buffer_size']
                    
        except Exception as e:
            # Store error for later logging
            self._config_load_error = f"Error loading global config: {e}. Using local recorder config only."
            print(self._config_load_error)
            
        return config
    
    def _get_default_config(self):
        """Return default configuration if config file is not available."""
        return {
            'recorder': {
                'camera_index': 0,
                'fps': 30,
                'resolution': {
                    'width': 1280,
                    'height': 720
                },
                'image_format': 'jpg',
                'output_dir': './frames',
                'buffer_size': 100,
                'keep_disk_copy': False
            }
        }
    
    def _setup_output_directory(self):
        """Create output directory if it doesn't exist."""
        # Get the recorder directory (where this file is located)
        recorder_dir = Path(__file__).parent.absolute()
        
        # Get the configured output directory from config
        config_output_dir = self.config['recorder']['output_dir']
        
        # If the path starts with './', it's already relative to the recorder directory
        if config_output_dir.startswith('./'):
            output_path = recorder_dir / config_output_dir[2:]
        # If the path is just a directory name without path separators, make it relative to recorder dir
        elif '/' not in config_output_dir and '\\' not in config_output_dir:
            output_path = recorder_dir / config_output_dir
        # If it starts with 'recorder/', remove that prefix and make it relative to recorder dir
        elif config_output_dir.startswith('recorder/'):
            output_path = recorder_dir / config_output_dir[9:]
        else:
            # Use as is (absolute path or other relative path)
            output_path = Path(config_output_dir)
        
        # Create the directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        self.output_folder = output_path
        self.logger.debug(f"Output directory: {self.output_folder}")
    
    def start_camera(self):
        """Initialize and start the camera."""
        try:
            camera_index = self.config['recorder']['camera_index']
            
            self.cap = cv2.VideoCapture(camera_index)
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FPS, self.config['recorder']['fps'])
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['recorder']['resolution']['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['recorder']['resolution']['height'])
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config['recorder']['buffer_size'])
            
            if not self.cap.isOpened():
                raise Exception(f"Failed to open camera at index {camera_index}")
            
            # Test capture a frame to verify camera is working
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                raise Exception("Camera opened but cannot capture frames")
            
            self.logger.info(f"Camera ready: {test_frame.shape[1]}x{test_frame.shape[0]}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop and release the camera."""
        if self.cap:
            self.cap.release()
            self.cap = None
            self.logger.debug("Camera stopped")
    
    def capture_frame(self):
        """Capture a single frame from the camera."""
        if not self.cap or not self.cap.isOpened():
            self.logger.error("Camera is not available")
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            self.logger.warning("Failed to capture frame")
            return None
        
        return frame
    
    def save_frame(self, frame):
        """Save a frame to disk in a date-based folder."""
        # Skip if disk copy is disabled
        if not self.config['recorder']['keep_disk_copy']:
            return False, None
        
        # Get current date for folder name
        current_date = datetime.now().strftime("%Y%m%d")
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"frame_{timestamp}.{self.config['recorder']['image_format']}"
        
        # Create date-based directory inside output folder
        date_folder = self.output_folder / current_date
        date_folder.mkdir(parents=True, exist_ok=True)
        
        # Create full filepath
        filepath = date_folder / filename
        
        # Save frame
        try:
            success = cv2.imwrite(str(filepath), frame)
            
            if success:
                self.frame_count += 1
                # Only log every 100 frames to reduce log volume
                if self.frame_count % 100 == 0:
                    self.logger.info(f"Saved {self.frame_count} frames in {current_date} folder")
                return True, str(filepath)
            else:
                self.logger.error(f"Failed to save frame: {filename}")
                return False, None
        except Exception as e:
            self.logger.error(f"Error saving frame: {e}")
            return False, None
    
    def start_recording(self):
        """Start the recording process."""
        if not self.start_camera():
            return False
        
        self.start_time = time.time()
        self.frame_count = 0
        self.logger.info("Recording started")
        
        try:
            while True:
                # Check if we should stop recording
                if self._should_stop_recording():
                    break
                
                # Capture and save frame
                frame = self.capture_frame()
                if frame is not None:
                    _, _ = self.save_frame(frame)  # Ignore return values here
                
                # Control frame rate
                time.sleep(1.0 / self.config['recorder']['fps'])
                
        except KeyboardInterrupt:
            self.logger.info("Recording stopped by user")
        except Exception as e:
            self.logger.error(f"Recording error: {e}")
        finally:
            self.stop_camera()
            self._print_recording_summary()
    
    def _should_stop_recording(self):
        """Check if recording should stop based on configuration."""
        # With the new config structure, recording continues indefinitely
        return False
        
    def start(self):
        """Start the recorder service."""
        try:
            if not self.start_camera():
                return False
            
            self.start_time = time.time()
            self.frame_count = 0
            self.running = True
            
            # Start frame capture thread
            self.capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
                name="FrameCapture"
            )
            self.capture_thread.start()
            
            self.logger.info("Recorder service started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start recorder service: {e}")
            return False
    
    def stop(self):
        """Stop the recorder service."""
        try:
            self.running = False
            
            # Wait for capture thread to finish
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=5.0)
                self.logger.info("Frame capture thread stopped")
            
            self.stop_camera()
            if self.start_time:
                self._print_recording_summary()
            self.logger.info("Recorder service stopped")
        except Exception as e:
            self.logger.error(f"Error stopping recorder service: {e}")
    
    def is_running(self):
        """Check if the recorder service is running."""
        # If keep_disk_copy is disabled, show as offline in UI
        if not self.config['recorder']['keep_disk_copy']:
            return False
        
        return self.running and self.cap is not None and self.cap.isOpened()
    
    def _capture_loop(self):
        """Main frame capture loop running in separate thread."""
        self.logger.info("Frame capture loop started")
        
        # Store frame buffer reference locally for thread safety
        frame_buffer = self.frame_buffer
        self.logger.info(f"Thread frame buffer: {frame_buffer}")
        self.logger.info(f"Thread frame buffer type: {type(frame_buffer)}")
        
        consecutive_failures = 0
        max_failures = 10
        
        while self.running:
            try:
                # Capture frame
                frame = self.capture_frame()
                if frame is not None:
                    consecutive_failures = 0  # Reset failure counter
                    
                    # Add ALL frames to buffer at recorder FPS rate
                    current_time = time.time()
                    
                    # Log detailed buffer status on first frame
                    if self.frame_count == 1:
                        self.logger.info(f"First frame captured - buffer info:")
                        self.logger.info(f"  Frame buffer: {frame_buffer is not None}")
                        self.logger.info(f"  Frame buffer type: {type(frame_buffer).__name__ if frame_buffer else 'None'}")
                        self.logger.info(f"  Recorder FPS: {self.config['recorder']['fps']}")
                    
                    # Add ALL frames to buffer at recorder FPS rate
                    if frame_buffer is not None:
                        try:
                            # Generate frame name for detection JSON
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            frame_name = f"frame_{timestamp}.{self.config['recorder']['image_format']}"
                            
                            # Add frame to buffer
                            success = frame_buffer.put(frame, frame_name)
                            if success:
                                # Update counters
                                self.buffer_frames_added += 1
                                
                                # Log buffer status periodically
                                if self.buffer_frames_added % 30 == 0:  # Log every 30 frames
                                    self.logger.info(f"Frame {self.frame_count} added to buffer (buffer frame #{self.buffer_frames_added})")
                                    self.logger.info(f"Buffer size: {len(frame_buffer)}/{frame_buffer.maxlen}")
                            else:
                                self.logger.warning(f"Failed to add frame {self.frame_count} to buffer (buffer full or timeout)")
                        except Exception as e:
                            self.logger.error(f"Failed to add frame {self.frame_count} to buffer: {e}")
                    elif frame_buffer is None:
                        self.logger.warning("No frame buffer available")

                    
                    # Save frame to disk if configured
                    success, filepath = self.save_frame(frame)
                    
                    # Don't increment frame_count here as it's already incremented in save_frame
                    
                    # Log progress every 30 frames
                    if self.frame_count % 30 == 0:
                        self.logger.info(f"Captured {self.frame_count} frames so far")
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        self.logger.error(f"Failed to capture frame {consecutive_failures} times in a row. Camera may not be available.")
                        break
                    self.logger.warning(f"Failed to capture frame (attempt {consecutive_failures})")
                
                # Control frame rate
                time.sleep(1.0 / self.config['recorder']['fps'])
                
            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)  # Brief pause on error
        
        self.logger.info("Frame capture loop stopped")

    def _print_recording_summary(self):
        """Print a summary of the recording session."""
        if not self.start_time:
            return
            
        duration = time.time() - self.start_time
        fps = self.frame_count / duration if duration > 0 else 0
        buffer_ratio = self.buffer_frames_added / max(1, self.frame_count) * 100 if self.frame_count > 0 else 0
        
        self.logger.info(f"Recording summary:")
        self.logger.info(f"  Duration: {duration:.2f} seconds")
        self.logger.info(f"  Frames captured: {self.frame_count}")
        self.logger.info(f"  Frames added to buffer: {self.buffer_frames_added} ({buffer_ratio:.1f}%)")
        self.logger.info(f"  Average capture FPS: {fps:.2f}")
        self.logger.info(f"  Target buffer FPS: {self.detection_fps}")
        
        # Calculate actual buffer FPS
        buffer_fps = self.buffer_frames_added / duration if duration > 0 else 0
        self.logger.info(f"  Actual buffer FPS: {buffer_fps:.2f}")
        
        # Report if buffer FPS matches target
        fps_match = abs(buffer_fps - self.detection_fps) / self.detection_fps * 100 if self.detection_fps > 0 else 0
        if fps_match <= 10:  # Within 10% of target
            self.logger.info(f"  Buffer FPS is within {fps_match:.1f}% of target - Good!")
        else:
            self.logger.warning(f"  Buffer FPS deviates by {fps_match:.1f}% from target")


def main():
    """Main function to run the recorder."""
    recorder = Recorder()
    
    print("Recorder initialized. Starting recording immediately...")
    print("Press Ctrl+C to stop recording and exit.")
    try:
        recorder.start_recording()
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
