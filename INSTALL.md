# Vision Service Installation Guide

This guide explains how to install and set up the Vision Service components.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd vision-service
```

### 2. Create and Activate a Virtual Environment (Recommended)

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Service-Specific Requirements

#### Recorder Service
```bash
pip install -r recorder/requirements.txt
```

#### Detector Service
```bash
pip install -r detector/requirements.txt
```

## Configuration

The vision service uses a centralized configuration approach:

- `config.yaml`: Main configuration file with settings for:
  - Buffer size
  - Recorder FPS
  - Detector FPS
  - Service startup delays
  - Performance settings
  - Logging configuration

- Service-specific configurations:
  - `recorder/config.yaml`: Camera settings, resolution, output directory
  - `detector/config.yaml`: Detection model settings, API configuration

## Running the Service

Start the complete vision service:

```bash
python run.py
```

This will:
1. Start the detector service
2. Start the recorder service
3. Initialize the coordinator
4. Begin processing frames

## Advanced Options

Run with custom configuration:
```bash
python run.py -c custom_config.yaml
```

Enable debug logging:
```bash
python run.py --debug
```

