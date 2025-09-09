#!/usr/bin/env python3
"""
Shared Utilities

Common utilities for logging, configuration, and helper functions.
"""

import os
import sys
import logging
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime
import traceback


def setup_logging(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path (None to disable file logging)
        format_string: Optional custom format string
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Set log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Default format with timestamp and level
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create formatter with ISO8601 timestamp format
    formatter = logging.Formatter(format_string, "%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file and log_file.lower() != 'none' and log_file != 'null':
        # No directory creation - assume file path is valid or use current directory
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (IOError, PermissionError) as e:
            # Log to console if file logging fails
            console_handler.setLevel(logging.WARNING)
            logger.warning(f"Could not set up log file '{log_file}': {e}")
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        json.JSONDecodeError: If JSON parsing fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load based on file extension
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Try YAML first, then JSON
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    config_path = Path(config_path)
    
    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save based on file extension
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    else:
        # Default to YAML
        with open(config_path.with_suffix('.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


def format_frame_info(frame_data: Any) -> str:
    """
    Format frame information for logging.
    
    Args:
        frame_data: Frame data object
        
    Returns:
        Formatted string with frame information
    """
    if hasattr(frame_data, 'shape'):
        # NumPy array or similar
        return f"Frame(shape={frame_data.shape}, dtype={getattr(frame_data, 'dtype', 'unknown')})"
    elif hasattr(frame_data, 'size'):
        # PIL Image or similar
        return f"Frame(size={frame_data.size}, mode={getattr(frame_data, 'mode', 'unknown')})"
    elif isinstance(frame_data, dict):
        # Dictionary with frame info
        return f"Frame({', '.join(f'{k}={v}' for k, v in frame_data.items())})"
    else:
        return f"Frame(type={type(frame_data).__name__})"


def retry_operation(
    operation,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry an operation with exponential backoff.
    
    Args:
        operation: Function to retry
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Result of the operation
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                time.sleep(current_delay)
                current_delay *= backoff_factor
            else:
                break
    
    raise last_exception


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(filename: str) -> str:
    """
    Convert filename to safe version by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    # Replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed"
    
    return filename


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def log_exception(logger: logging.Logger, message: str = "An error occurred") -> None:
    """
    Log an exception with full traceback.
    
    Args:
        logger: Logger instance
        message: Custom error message
    """
    logger.error(f"{message}: {traceback.format_exc()}")


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate configuration has required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required configuration keys
        
    Returns:
        True if valid, False otherwise
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        logging.error(f"Missing required configuration keys: {missing_keys}")
        return False
    return True


# Import time module for retry_operation
import time
