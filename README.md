# Vision Service

A real-time computer vision service for object detection using RT-DETR v2 ONNX model. The service consists of a camera recorder, object detector, and web UI for monitoring and manual detection.

## Architecture

The service is composed of three main components:

- **Recorder**: Captures frames from camera and stores them in a buffer
- **Detector**: Performs object detection using RT-DETR v2 ONNX model
- **Coordinator**: Orchestrates the recorder and detector services
- **UI**: Web interface for monitoring and manual detection

## Installation

### Prerequisites

- Python 3.8 or higher
- Camera/webcam access
- Windows/Linux/macOS

### 1. Clone the Repository

```bash
git clone <repository-url>
cd vision-service
```

### 2. Install Dependencies

Install dependencies for each component:

```bash
# Install coordinator dependencies
pip install -r requirements.txt

# Install detector dependencies
pip install -r detector/requirements.txt

# Install UI dependencies
pip install -r ui/requirements.txt
```

### 3. Download ONNX Model

Place your RT-DETR v2 ONNX model file in the `detector/` directory as `model.onnx`.

## Configuration

### Detector Configuration

Edit `detector/config.yaml` to configure the detector:

```yaml
model:
  onnx_file: "model.onnx"
  img_size: 640
  conf_thres: 0.6
  device: "cpu"  # or "cuda" for GPU

api:
  host: "0.0.0.0"
  port: 8000
  reload: false

paths:
  categories_file: "categories.yaml"
  default_image: "./default/default_img.jpg"
  output_directory: "output"

save_results:
  enabled: true
  output_dir: "output"
```

### Recorder Configuration

Edit `recorder/config.yaml` to configure the recorder:

```yaml
camera:
  source: 0  # Camera index (0 for default camera)
  width: 640
  height: 480

fps: 30

buffer:
  max_size: 100
  timeout: 5.0

# Disk storage settings
keep_disk_copy: false  # Set to false to disable disk storage
```

**Note**: When `keep_disk_copy: false` is set, the recorder status in the UI will show as **OFFLINE** even though the recorder is actually running. This is because the UI checks for active frame processing, which may be affected by the disk storage setting.

### Coordinator Configuration

Edit `config.yaml` to configure the coordinator:

```yaml
recorder:
  enabled: true
  fps: 30

detector:
  enabled: true
  api_port: 8000

ui:
  enabled: true
  port: 7860
```

## Running the Services

### Run All Services

Start the complete vision service:

```bash
python run.py
```

This will start:
- Recorder service (camera capture)
- Detector service (ONNX inference API)
- Coordinator (orchestration)
- Web UI (monitoring interface)

### Run with UI Only

Start the service with web UI only (no camera recording):

```bash
python run.py --ui
```

This will start:
- Detector service (ONNX inference API)
- Web UI (monitoring interface)
- Manual detection mode (no automatic camera capture)

## Usage

### Web Interface

1. Open your browser and go to `http://localhost:7860`
2. The interface shows:
   - **Camera Feed**: Live camera stream
   - **System Status**: Recorder and detector status
   - **Manual Detection**: Click "Run Detection" to trigger inference
   - **Detection Results**: Shows detected objects with confidence scores

### API Endpoints

#### Detection Endpoints

- **`POST /detect`**: Main detection endpoint
  - Upload an image file for detection
  - Or send empty request to use latest buffer frame
  - Falls back to default image if buffer is empty

- **`POST /detect-latest`**: Detect on latest buffer frame
  - No file upload required
  - Uses the freshest frame from camera buffer
  - Returns 404 if no frames available

- **`GET /detect-default`**: Detect on default test image
  - Uses the default image for testing
  - Useful for API testing without camera

#### Example API Usage

```bash
# Detect on uploaded image
curl -X POST "http://localhost:8000/detect" \
  -F "file=@image.jpg"

# Detect on latest camera frame
curl -X POST "http://localhost:8000/detect-latest"

# Detect on default image
curl -X GET "http://localhost:8000/detect-default"
```

### Response Format

All detection endpoints return JSON in this format:

```json
{
  "detections": [
    {
      "class": "person",
      "confidence": 0.95,
      "bbox": [100, 200, 300, 400]
    }
  ],
  "inference_time": 0.045,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Features

### Real-time Processing
- Continuous camera capture at configurable FPS
- Frame buffering with automatic cleanup
- Non-blocking inference processing

### Manual Detection
- On-demand detection via web UI or API
- Uses freshest available frame
- No need to upload files

### Buffer Management
- Automatic buffer cleanup when full
- Keeps newest frames, drops oldest
- Configurable buffer size and timeout

### Error Handling
- Graceful fallback to default image
- Comprehensive error logging
- HTTP status codes for different error types
