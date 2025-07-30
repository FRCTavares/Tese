#!/usr/bin/env python3
"""
Web-based Real-Time Object Detection Viewer
Streams detection results to a web browser for easy viewing.
"""

import cv2
import numpy as np
import time
import sys
import logging
from pathlib import Path
from flask import Flask, Response, render_template_string
import threading
import queue
from typing import Optional

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from camera import RealSenseStreamer
    REALSENSE_AVAILABLE = True
except ImportError as e:
    print(f"RealSense not available: {e}")
    print("Using mock camera for testing...")
    from mock_camera import MockRealSenseStreamer as RealSenseStreamer
    REALSENSE_AVAILABLE = False

from detector import Detector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for frame sharing
frame_queue = queue.Queue(maxsize=2)
stats_queue = queue.Queue(maxsize=1)

# HTML template for the web viewer
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Pi 5 Real-Time Object Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            text-align: center;
        }
        .video-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .stats {
            display: flex;
            justify-content: space-around;
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stat-item {
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #2196F3;
        }
        .stat-label {
            font-size: 14px;
            color: #666;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 5px;
        }
        .status {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .status.connected {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
    </style>
    <script>
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('detections').textContent = data.detections;
                    document.getElementById('inference-time').textContent = (data.inference_time * 1000).toFixed(0);
                })
                .catch(error => console.log('Stats update failed:', error));
        }
        
        setInterval(updateStats, 1000);  // Update every second
        window.onload = updateStats;
    </script>
</head>
<body>
    <div class="container">
        <h1>🎯 Raspberry Pi 5 Real-Time Object Detection</h1>
        
        <div class="status connected">
            <strong>✅ Camera:</strong> Intel RealSense D435 | 
            <strong>Model:</strong> YOLOv8-nano | 
            <strong>Status:</strong> Live Streaming
        </div>
        
        <div class="video-container">
            <h2>Live Detection Feed</h2>
            <img src="{{ url_for('video_feed') }}" alt="Live Detection Stream">
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-value" id="fps">0.0</div>
                <div class="stat-label">FPS</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="detections">0</div>
                <div class="stat-label">Detections</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="inference-time">0</div>
                <div class="stat-label">Inference (ms)</div>
            </div>
        </div>
        
        <p style="margin-top: 20px; color: #666;">
            🔄 Stream updates automatically | 
            🎨 Bounding boxes show detected objects | 
            📊 Statistics update every second
        </p>
    </div>
</body>
</html>
"""

class WebDetectionPipeline:
    """Real-time detection pipeline optimized for web streaming."""
    
    def __init__(self, width=424, height=240, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        
        self.camera: Optional[RealSenseStreamer] = None
        self.detector: Optional[Detector] = None
        self.running = False
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = 0.0
        self.current_fps = 0.0
        self.current_detections = 0
        self.current_inference_time = 0.0
        
    def initialize(self):
        """Initialize camera and detector."""
        logger.info("Initializing web detection pipeline...")
        
        # Initialize camera
        self.camera = RealSenseStreamer(
            width=self.width,
            height=self.height,
            fps=self.fps,
            enable_depth=False
        )
        
        if not self.camera.start():
            logger.error("Failed to start camera")
            return False
            
        # Initialize detector
        self.detector = Detector(
            model_path="yolov8n.pt",
            confidence_threshold=0.5
        )
        
        if not self.detector.load_model():
            logger.error("Failed to load detection model")
            return False
            
        logger.info("Web detection pipeline initialized successfully")
        return True
        
    def run(self):
        """Run the detection pipeline in a separate thread."""
        self.running = True
        self.start_time = time.time()
        
        logger.info("Starting web detection pipeline...")
        
        while self.running:
            try:
                # Get frame from camera
                frame_data = self.camera.get_frame()
                if frame_data is None:
                    continue
                    
                color_frame, _ = frame_data
                
                # Run detection
                detections = self.detector.detect(color_frame)
                
                # Annotate frame
                annotated_frame = self.detector.annotate_frame(color_frame, detections)
                
                # Update performance stats
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                self.current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                self.current_detections = len(detections)
                
                # Get detector stats
                detector_stats = self.detector.get_performance_stats()
                self.current_inference_time = detector_stats.get('avg_inference_time', 0)
                
                # Encode frame for web streaming
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                # Update frame queue (non-blocking)
                try:
                    frame_queue.put_nowait(frame_bytes)
                except queue.Full:
                    try:
                        frame_queue.get_nowait()  # Remove old frame
                        frame_queue.put_nowait(frame_bytes)  # Add new frame
                    except queue.Empty:
                        pass
                
                # Update stats queue
                stats = {
                    'fps': self.current_fps,
                    'detections': self.current_detections,
                    'inference_time': self.current_inference_time
                }
                try:
                    stats_queue.put_nowait(stats)
                except queue.Full:
                    try:
                        stats_queue.get_nowait()
                        stats_queue.put_nowait(stats)
                    except queue.Empty:
                        pass
                        
            except Exception as e:
                logger.error(f"Error in detection pipeline: {e}")
                
        logger.info("Web detection pipeline stopped")
        
    def stop(self):
        """Stop the detection pipeline."""
        self.running = False
        if self.camera:
            self.camera.stop()

# Global pipeline instance
pipeline = WebDetectionPipeline()

@app.route('/')
def index():
    """Serve the main web page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    def generate():
        while True:
            try:
                # Get latest frame
                frame_bytes = frame_queue.get(timeout=1.0)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except queue.Empty:
                # Send a placeholder frame if no data available
                placeholder = np.zeros((240, 424, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for frames...", (50, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', placeholder)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    """Get current performance statistics."""
    try:
        stats = stats_queue.get_nowait()
        return stats
    except queue.Empty:
        return {
            'fps': 0.0,
            'detections': 0,
            'inference_time': 0.0
        }

def main():
    """Main entry point for web viewer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Web-based Real-Time Object Detection Viewer")
    parser.add_argument('--width', type=int, default=424, help='Camera width')
    parser.add_argument('--height', type=int, default=240, help='Camera height')
    parser.add_argument('--fps', type=int, default=30, help='Camera FPS')
    parser.add_argument('--port', type=int, default=5000, help='Web server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web server host')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    global pipeline
    pipeline = WebDetectionPipeline(args.width, args.height, args.fps)
    
    if not pipeline.initialize():
        logger.error("Failed to initialize pipeline")
        return
    
    # Start detection in background thread
    detection_thread = threading.Thread(target=pipeline.run, daemon=True)
    detection_thread.start()
    
    try:
        print(f"\n🌐 Web Viewer Starting...")
        print(f"📡 Access the live stream at: http://localhost:{args.port}")
        print(f"🎥 Camera: {args.width}x{args.height} @ {args.fps}fps")
        print(f"🔥 Press Ctrl+C to stop\n")
        
        # Start web server
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping web viewer...")
    finally:
        pipeline.stop()

if __name__ == '__main__':
    main()
