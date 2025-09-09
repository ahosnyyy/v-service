import gradio as gr
import datetime
import time
import requests
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coordinator.coordinator import VisionCoordinator

def create_ui(coordinator: 'VisionCoordinator'):
    
    # Detector API configuration
    DETECTOR_API_URL = "http://localhost:8000"
    
    def call_detector_api():
        """Call the detector API to get latest detection results."""
        try:
            response = requests.get(f"{DETECTOR_API_URL}/detect-latest", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Detector API error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error calling detector API: {e}")
            return None
    
    def get_detector_buffer_status():
        """Get detector buffer status."""
        try:
            response = requests.get(f"{DETECTOR_API_URL}/buffer-status", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            print(f"Error getting buffer status: {e}")
            return None
    
    def get_status_updates():
        status = coordinator.get_status()
        
        # Get actual recorder FPS from config or status
        recorder_fps = 0
        if coordinator.recorder:
            try:
                # First try to get from global config
                global_config = coordinator.config if hasattr(coordinator, 'config') else {}
                recorder_config = global_config.get('recorder', {})
                recorder_fps = recorder_config.get('fps', 0)
                
                # If not found, try recorder's own config
                if recorder_fps == 0 and hasattr(coordinator.recorder, 'config'):
                    recorder_fps = coordinator.recorder.config.get('fps', 0)
                
                # Last resort: try recorder's fps attribute
                if recorder_fps == 0 and hasattr(coordinator.recorder, 'fps'):
                    recorder_fps = coordinator.recorder.fps
            except:
                recorder_fps = 0
        
        detector_fps = status.get("detection_fps", 0)
        
        # Get buffer statistics from both coordinator and detector
        coordinator_buffer_stats = {}
        if coordinator.frame_buffer:
            try:
                coordinator_buffer_stats = coordinator.frame_buffer.get_stats()
            except:
                coordinator_buffer_stats = {}
        
        # Get detector buffer status
        detector_buffer_status = get_detector_buffer_status()
        detector_buffer_stats = detector_buffer_status.get('buffer_stats', {}) if detector_buffer_status else {}

        recorder_active = status.get('recorder_active')
        detector_active = status.get('detector_active')
        
        recorder_status_md = f"<div style='display: flex; justify-content: space-between; align-items: center;'>Rate: {recorder_fps} FPS <span style='background-color: {'#22c55e' if recorder_active else '#ef4444'}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.8em;'>{'ACTIVE' if recorder_active else 'OFFLINE'}</span></div>"
        detector_status_md = f"<div style='display: flex; justify-content: space-between; align-items: center;'>Rate: {detector_fps} FPS <span style='background-color: {'#22c55e' if detector_active else '#ef4444'}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.8em;'>{'ACTIVE' if detector_active else 'OFFLINE'}</span></div>"
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Include buffer info in subtitle
        buffer_info = ""
        if coordinator_buffer_stats or detector_buffer_stats:
            coord_size = coordinator_buffer_stats.get('current_size', 0)
            coord_max = coordinator_buffer_stats.get('max_size', 0)
            coord_util = coordinator_buffer_stats.get('utilization', 0) * 100
            
            det_size = detector_buffer_stats.get('current_size', 0)
            det_max = detector_buffer_stats.get('max_size', 0)
            det_util = detector_buffer_stats.get('utilization', 0) * 100
            
            buffer_info = f" | Coord Buffer: {coord_size}/{coord_max} ({coord_util:.1f}%) | Detector Buffer: {det_size}/{det_max} ({det_util:.1f}%)"
        
        subtitle = f"<div style='text-align: right;'>Real-time status as of: {timestamp}</div>"

        return subtitle, recorder_status_md, detector_status_md

    def get_detection_frame():
        # Since we disabled automatic detection, we need to get frames from coordinator's buffer
        # The coordinator still has frames, but no automatic processing
        if hasattr(coordinator, 'last_detection_frame') and coordinator.last_detection_frame is not None:
            frame = coordinator.last_detection_frame.copy()
            
            if isinstance(frame, np.ndarray):
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Convert BGR to RGB
                    frame = frame[:, :, ::-1]
                return frame
        
        # Try to get latest frame from coordinator's buffer
        if coordinator.frame_buffer and not coordinator.frame_buffer.empty():
            try:
                frame_data = coordinator.frame_buffer.get_latest(block=False)
                if frame_data is not None:
                    if isinstance(frame_data, tuple) and len(frame_data) == 2:
                        frame = frame_data[0]
                    else:
                        frame = frame_data
                    
                    if isinstance(frame, np.ndarray):
                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                            frame = frame[:, :, ::-1]
                        return frame
            except:
                pass
        
        return None # Return None if no frame is available

    def get_detection_summary():
        # Call detector API to get latest detection results
        try:
            detection_result = call_detector_api()
            if detection_result and 'detections' in detection_result:
                detections = detection_result['detections']
                
                if detections:
                    # Create styled tags for each detection with confidence
                    tags = []
                    for det in detections:
                        class_name = det.get('class') or det.get('class_name') or det.get('label') or det.get('name') or 'unknown'
                        confidence = det.get('confidence', 0)
                        conf_percent = int(confidence * 100) if confidence <= 1 else int(confidence)
                        
                        # Create styled tag similar to status tags
                        tag = f"<span style='background-color: #3b82f6; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; margin-right: 4px; display: inline-block;'>{class_name} ({conf_percent}%)</span>"
                        tags.append(tag)
                    
                    summary_md = " ".join(tags)
                    return summary_md
                else:
                    # Inference was done but no detections found
                    return "<span style='color: #6b7280; font-style: italic;'>Nothing detected</span>"
            
            # Return empty string if no inference has been done yet
            return ""
            
        except Exception as e:
            return f"**Status:** Error retrieving detection data: {str(e)}"

    with gr.Blocks() as app:
        # Header Section
        with gr.Row():
            gr.Markdown("## Detection Service Monitor")
            subtitle_text = gr.HTML("*Real-time status...*")

        # Main Content Layout
        with gr.Row():
            # Left column - Detection frame only
            with gr.Column(scale=3):
                with gr.Group():
                    gr.Markdown("**📷 Live Detection Feed**")
                    detection_image = gr.Image(
                        label="Clothing", 
                        type="pil", 
                        interactive=False,
                        height=500
                    )
            
            # Right column - System status
            with gr.Column(scale=1):
                with gr.Group():                    
                    # Service status cards
                    with gr.Group():
                        gr.Markdown("**📹 Recorder**")
                        recorder_status = gr.HTML(
                            value="<div style='display: flex; justify-content: space-between; align-items: center;'>Rate: 30 FPS <span style='background-color: #22c55e; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.8em;'>ACTIVE</span></div>"
                        )
                    
                    # Separation line
                    gr.HTML("<hr style='border: none; border-top: 1px solid #e5e7eb; margin: 10px 0;'>")
                    
                    with gr.Group():
                        gr.Markdown("**👁️ Detector**")
                        detector_status = gr.HTML(
                            value="<div style='display: flex; justify-content: space-between; align-items: center;'>Rate: 1 FPS <span style='background-color: #22c55e; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.8em;'>ACTIVE</span></div>"
                        )
                    
                    # Control buttons
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Refresh Status", variant="primary", size="sm")
                        detect_btn = gr.Button("🔍 Run Detection", variant="secondary", size="sm")
                
                # Detection summary section
                with gr.Group():
                    gr.Markdown("**👕 Detections**")
                    detection_summary = gr.HTML("")
        
        # Set up event handlers
        refresh_btn.click(
            fn=get_status_updates,
            inputs=None,
            outputs=[subtitle_text, recorder_status, detector_status]
        )
        
        refresh_btn.click(
            fn=get_detection_frame,
            inputs=None,
            outputs=[detection_image]
        )
        
        # Manual detection button
        detect_btn.click(
            fn=get_detection_summary,
            inputs=None,
            outputs=[detection_summary]
        )
        
        detect_btn.click(
            fn=get_detection_frame,
            inputs=None,
            outputs=[detection_image]
        )
        
        # Auto-refresh detection frame every 2 seconds (manual detection mode)
        detection_timer = gr.Timer(2)
        detection_timer.tick(
            fn=get_detection_frame,
            inputs=None,
            outputs=[detection_image]
        )
        
        # Auto-refresh status every 5 seconds
        status_timer = gr.Timer(5)
        status_timer.tick(
            fn=get_status_updates,
            inputs=None,
            outputs=[subtitle_text, recorder_status, detector_status]
        )
        
        # Auto-refresh on page load
        app.load(
            fn=get_status_updates,
            inputs=None,
            outputs=[subtitle_text, recorder_status, detector_status]
        )
        
        app.load(
            fn=get_detection_summary,
            inputs=None,
            outputs=[detection_summary]
        )

    return app

if __name__ == "__main__":
    # Mock coordinator for standalone testing
    class MockCoordinator:
        def get_status(self):
            return {'running': True, 'recorder_active': True, 'detector_active': True, 'detection_fps': 10}
        @property
        def recorder(self):
            class MockRecorder:
                fps = 30
            return MockRecorder()
        @property
        def frame_buffer(self):
            return None

    ui = create_ui(MockCoordinator())
    ui.launch()
