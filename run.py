#!/usr/bin/env python3
"""
Vision Service - Main Entry Point

Loads configurations and starts the coordinator service.
"""

import sys
import signal
import argparse
import subprocess
import time
import os
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from shared.utils import setup_logging, load_config, log_exception
from coordinator import VisionCoordinator


def setup_signal_handlers(coordinator: VisionCoordinator) -> None:
    """Set up signal handlers for graceful shutdown."""
    
    def signal_handler(signum, frame):
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        coordinator.stop()
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    # Windows-specific signals
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)


def start_detection_service(logger) -> bool:
    """Start the detection service as a background process."""
    try:
        logger.info("Detection service starting...")
        
        # Check if detector service is already running
        try:
            import requests
            response = requests.get("http://localhost:8000/", timeout=2)
            if response.status_code == 200:
                logger.info("Detection service already running")
                return True
        except:
            pass
        
        # Start detector service in background
        detector_script = Path("detector/detector.py")
        if not detector_script.exists():
            logger.error(f"Detector script not found: {detector_script}")
            return False
        
        # Start the service
        process = subprocess.Popen(
            [sys.executable, str(detector_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=project_root
        )
        
        # Wait for service to start
        max_wait = 30  # seconds
        wait_time = 0
        
        while wait_time < max_wait:
            try:
                response = requests.get("http://localhost:8000/", timeout=2)
                if response.status_code == 200:
                    logger.info("Detection service ready")
                    return True
            except:
                pass
            
            time.sleep(1)
            wait_time += 1
            
            if wait_time % 5 == 0:
                logger.info(f"Waiting for detector ({wait_time}s)")
        
        # If we get here, service didn't start
        logger.error("Detection service failed to start within timeout")
        process.terminate()
        return False
        
    except Exception as e:
        logger.error(f"Failed to start detection service: {e}")
        return False


def validate_configs(config: dict) -> bool:
    """Validate main configuration for coordinator."""
    # Check for required keys
    required_keys = ["buffer_size"]
    
    # Check for either detector.fps or the legacy detection_interval
    detector_config = config.get("detector", {})
    if "fps" not in detector_config and "detection_interval" not in config:
        print("Missing required configuration: either detector.fps or detection_interval must be present")
        return False
    
    # Check main config for other required keys
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        print(f"Missing required configuration keys: {missing_keys}")
        return False
    
    # Check that service config files exist (each service manages its own config)
    recorder_config = Path("recorder/config.yaml")
    detector_config = Path("detector/config.yaml")
    
    if not recorder_config.exists():
        print(f"Recorder config not found: {recorder_config}")
        return False
    
    if not detector_config.exists():
        print(f"Detector config not found: {detector_config}")
        return False
    
    print("Configuration validation passed")
    return True


def main():
    """Main entry point for the vision service."""
    parser = argparse.ArgumentParser(
        description="Vision Service - Computer Vision Recording and Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                           # Use default config.yaml, auto-start detector
  python run.py -c custom_config.yaml    # Use custom config file
  python run.py --debug                  # Enable debug logging
  python run.py --log-file logs/app.log  # Custom log file
        """
    )
    
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--log-file",
        help="Path to log file (default: logs/vision_service.log)"
    )
    
    
    parser.add_argument(
        "--version",
        action="version",
        version="Vision Service v1.0.0"
    )

    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the Gradio monitoring UI"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    log_file = None  # Disable log file
    
    # Use proper format with timestamps and levels
    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger = setup_logging("vision_service", log_level, log_file, format_string)
    logger.info("Starting Vision Service")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Validate configuration
        if not validate_configs(config):
            logger.error("Configuration validation failed")
            return 1
        
        # Start detection service
        if not start_detection_service(logger):
            logger.error("Failed to start detection service")
            return 1
        
        # Create and start coordinator
        coordinator = VisionCoordinator(args.config)
        
        # Setup signal handlers for graceful shutdown
        setup_signal_handlers(coordinator)

        # Launch UI if requested
        if args.ui:
            try:
                from ui.app import create_ui
                import threading
                
                logger.info("Launching Gradio UI...")
                
                # The UI needs access to the coordinator to get data
                ui_app = create_ui(coordinator) 
                
                # Run Gradio in a separate thread
                ui_thread = threading.Thread(
                    target=lambda: ui_app.launch(server_name="0.0.0.0", server_port=7860, share=False),
                    daemon=True
                )
                ui_thread.start()
                logger.info("UI is running at http://localhost:7860")

            except ImportError:
                logger.error("UI dependencies are not installed. Please run:")
                logger.error("pip install -r ui/requirements.txt")
            except Exception as e:
                logger.error(f"Failed to launch UI: {e}")
        
        # Start the service
        if not coordinator.start():
            logger.error("Failed to start Vision Service")
            return 1
        
        logger.info("Vision Service ready - Press Ctrl+C to stop")
        
        # Keep the service running
        try:
            while coordinator.is_running():
                # Get status every 30 seconds
                import time
                time.sleep(30)
                
                # Log status
                status = coordinator.get_status()
                logger.info(f"Status: {status}")
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            log_exception(logger, "Unexpected error in main loop")
            return 1
        
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        return 1
    except Exception as e:
        log_exception(logger, "Failed to start Vision Service")
        return 1
    finally:
        # Ensure clean shutdown
        logger.info("Shutting down Vision Service...")
        if 'coordinator' in locals():
            coordinator.stop()
        logger.info("Vision Service stopped")
        logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
