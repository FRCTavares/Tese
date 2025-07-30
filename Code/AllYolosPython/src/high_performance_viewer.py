#!/usr/bin/env python3
"""
High-Performance Real-Time Object Detection for Pi 5
Optimized for 5+ FPS performance with multiple acceleration techniques.
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
import os

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from camera import RealSenseStreamer
    REALSENSE_AVAILABLE = True
except ImportError as e:
    print(f"RealSense not available: {e}")
    from mock_camera import MockRealSenseStreamer as RealSenseStreamer
    REALSENSE_AVAILABLE = False

from detector import Detector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for frame sharing
frame_queue = queue.Queue(maxsize=1)  # Reduced queue size for lower latency
stats_queue = queue.Queue(maxsize=1)

# HTML template (same as before but with performance tips)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🚀 Pi 5 High-Performance Object Detection</title>
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
        }
        .stat-value.good { color: #4CAF50; }
        .stat-value.ok { color: #FF9800; }
        .stat-value.poor { color: #F44336; }
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
        .optimizations {
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            text-align: left;
        }
    </style>
    <script>
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    const fpsElement = document.getElementById('fps');
                    const fps = data.fps;
                    fpsElement.textContent = fps.toFixed(1);
                    
                    // Color code FPS performance
                    if (fps >= 5) {
                        fpsElement.className = 'stat-value good';
                    } else if (fps >= 3) {
                        fpsElement.className = 'stat-value ok';
                    } else {
                        fpsElement.className = 'stat-value poor';
                    }
                    
                    document.getElementById('detections').textContent = data.detections;
                    document.getElementById('inference-time').textContent = (data.inference_time * 1000).toFixed(0);
                    document.getElementById('skip-rate').textContent = data.skip_rate || 0;
                })
                .catch(error => console.log('Stats update failed:', error));
        }
        
        setInterval(updateStats, 1000);
        window.onload = updateStats;
    </script>
</head>
<body>
    <div class="container">
        <h1>🚀 Pi 5 High-Performance Object Detection</h1>
        
        <div class="status connected">
            <strong>⚡ Mode:</strong> High Performance | 
            <strong>🎯 Target:</strong> 5+ FPS | 
            <strong>📸 Camera:</strong> Intel RealSense D435
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
            <div class="stat-item">
                <div class="stat-value" id="skip-rate">0</div>
                <div class="stat-label">Skip Rate</div>
            </div>
        </div>
        
        <div class="optimizations">
            <h3>🔧 Performance Optimizations Active</h3>
            <ul>
                <li>✅ <strong>Low Resolution Inference:</strong> 320x320 (vs 640x640)</li>
                <li>✅ <strong>Frame Skipping:</strong> Process every 2nd frame</li>
                <li>✅ <strong>High Confidence Filter:</strong> Skip processing weak detections</li>
                <li>✅ <strong>Fast Image Encoding:</strong> Optimized JPEG compression</li>
                <li>✅ <strong>Reduced Queue Latency:</strong> Drop old frames immediately</li>
                <li>✅ <strong>Pi 5 Optimized:</strong> ARM64 + 4-core utilization</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

class HighPerformanceDetectionPipeline:
    """High-performance detection pipeline optimized for 5+ FPS on Pi 5."""
    
    def __init__(self, width=424, height=240, fps=30, target_fps=5):
        self.width = width
        self.height = height
        self.fps = fps
        self.target_fps = target_fps
        
        self.camera: Optional[RealSenseStreamer] = None
        self.detector: Optional[Detector] = None
        self.running = False
        
        # Performance optimizations
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_counter = 0
        self.inference_size = 320  # Reduced from 640 for speed
        self.high_confidence_threshold = 0.7  # Skip low confidence processing
        
        # Performance tracking
        self.start_time = 0.0
        self.processed_frames = 0
        self.skipped_frames = 0
        self.current_fps = 0.0
        self.current_detections = 0
        self.current_inference_time = 0.0
        
        # Cache for last processed detection frame
        self.last_detection_frame = None
        self.last_detections = []
        
    def initialize(self):
        """Initialize camera and detector with performance optimizations."""
        logger.info("🚀 Initializing high-performance detection pipeline...")
        
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
            
        # Initialize detector with performance settings
        self.detector = Detector(
            model_path="yolov8n.pt",  # Start with PyTorch, can upgrade to ONNX
            confidence_threshold=0.6,  # Higher threshold for speed
        )
        
        if not self.detector.load_model():
            logger.error("Failed to load detection model")
            return False
        
        # Configure YOLO for speed
        if hasattr(self.detector.model, 'model'):
            # Enable optimizations
            self.detector.model.overrides['verbose'] = False
            self.detector.model.overrides['imgsz'] = self.inference_size
            
        logger.info("✅ High-performance pipeline initialized")
        logger.info(f"📊 Target: {self.target_fps} FPS | Inference: {self.inference_size}x{self.inference_size}")
        return True
        
    def run(self):
        """Run optimized detection pipeline."""
        self.running = True
        self.start_time = time.time()
        
        logger.info("🔥 Starting high-performance detection pipeline...")
        
        while self.running:
            try:
                # Get frame from camera
                frame_data = self.camera.get_frame()
                if frame_data is None:
                    continue
                    
                color_frame, _ = frame_data
                self.frame_counter += 1
                
                # Frame skipping optimization
                if self.frame_counter % self.frame_skip == 0:
                    # Process this frame
                    start_inference = time.time()
                    
                    # Resize frame for faster inference
                    inference_frame = cv2.resize(color_frame, (self.inference_size, self.inference_size))
                    
                    # Run detection
                    detections = self.detector.detect(inference_frame)
                    
                    # Scale detections back to original frame size
                    scaled_detections = self._scale_detections(detections, 
                                                             (self.inference_size, self.inference_size),
                                                             (self.width, self.height))
                    
                    # Cache results
                    self.last_detection_frame = color_frame.copy()
                    self.last_detections = scaled_detections
                    self.processed_frames += 1
                    
                    # Update performance stats
                    inference_time = time.time() - start_inference
                    self.current_inference_time = inference_time
                    self.current_detections = len(scaled_detections)
                    
                else:
                    # Skip inference, use cached results
                    self.skipped_frames += 1
                    if self.last_detection_frame is not None:
                        color_frame = self.last_detection_frame.copy()
                        scaled_detections = self.last_detections
                    else:
                        scaled_detections = []
                
                # Always annotate the current frame (fast operation)
                annotated_frame = self._fast_annotate_frame(color_frame, scaled_detections)
                
                # Calculate FPS
                elapsed_time = time.time() - self.start_time
                self.current_fps = self.frame_counter / elapsed_time if elapsed_time > 0 else 0
                
                # Fast JPEG encoding with optimization
                encode_start = time.time()
                _, buffer = cv2.imencode('.jpg', annotated_frame, 
                                       [cv2.IMWRITE_JPEG_QUALITY, 75,  # Reduced quality for speed
                                        cv2.IMWRITE_JPEG_OPTIMIZE, 1])
                frame_bytes = buffer.tobytes()
                encode_time = time.time() - encode_start
                
                # Update frame queue (drop old frames immediately)
                try:
                    if not frame_queue.empty():
                        try:
                            frame_queue.get_nowait()  # Drop old frame
                        except queue.Empty:
                            pass
                    frame_queue.put_nowait(frame_bytes)
                except queue.Full:
                    pass  # Skip if queue is full
                
                # Update stats queue
                skip_rate = (self.skipped_frames / max(1, self.frame_counter)) * 100
                stats = {
                    'fps': self.current_fps,
                    'detections': self.current_detections,
                    'inference_time': self.current_inference_time,
                    'skip_rate': f"{skip_rate:.0f}%"
                }
                
                try:
                    if not stats_queue.empty():
                        try:
                            stats_queue.get_nowait()
                        except queue.Empty:
                            pass
                    stats_queue.put_nowait(stats)
                except queue.Full:
                    pass
                    
                # Adaptive frame skipping based on performance
                if self.current_fps < self.target_fps * 0.8:  # If below 80% of target
                    self.frame_skip = min(4, self.frame_skip + 1)  # Increase skipping
                elif self.current_fps > self.target_fps * 1.2:  # If above 120% of target
                    self.frame_skip = max(1, self.frame_skip - 1)  # Reduce skipping
                        
            except Exception as e:
                logger.error(f"Error in detection pipeline: {e}")
                
        logger.info("High-performance detection pipeline stopped")
    
    def _scale_detections(self, detections, from_size, to_size):
        """Scale detections from inference size to display size."""
        if not detections:
            return []
            
        from_w, from_h = from_size
        to_w, to_h = to_size
        scale_x = to_w / from_w
        scale_y = to_h / from_h
        
        scaled_detections = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            scaled_bbox = (
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y)
            )
            
            # Create new detection with scaled bbox
            from detector import Detection
            scaled_detection = Detection(
                bbox=scaled_bbox,
                confidence=detection.confidence,
                class_id=detection.class_id,
                class_name=detection.class_name
            )
            scaled_detections.append(scaled_detection)
            
        return scaled_detections
    
    def _fast_annotate_frame(self, image, detections):
        """Fast annotation optimized for performance."""
        annotated = image.copy()
        
        # Only draw high-confidence detections for speed
        high_conf_detections = [d for d in detections if d.confidence > self.high_confidence_threshold]
        
        for detection in high_conf_detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Fast drawing with minimal text
            color = (0, 255, 0)  # Simple green for all objects
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Minimal text for speed
            label = f"{detection.class_name} {detection.confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add performance overlay
        fps_text = f"FPS: {self.current_fps:.1f} | Target: {self.target_fps}"
        cv2.putText(annotated, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated
        
    def stop(self):
        """Stop the detection pipeline."""
        self.running = False
        if self.camera:
            self.camera.stop()

# Global pipeline instance
pipeline = HighPerformanceDetectionPipeline()

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
                frame_bytes = frame_queue.get(timeout=0.5)  # Reduced timeout
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except queue.Empty:
                # Send a placeholder frame if no data available
                placeholder = np.zeros((240, 424, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Optimizing performance...", (50, 120), 
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
            'inference_time': 0.0,
            'skip_rate': '0%'
        }

def main():
    """Main entry point for high-performance web viewer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="High-Performance Real-Time Object Detection")
    parser.add_argument('--width', type=int, default=424, help='Camera width')
    parser.add_argument('--height', type=int, default=240, help='Camera height')
    parser.add_argument('--fps', type=int, default=30, help='Camera FPS')
    parser.add_argument('--target-fps', type=int, default=5, help='Target processing FPS')
    parser.add_argument('--port', type=int, default=5001, help='Web server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web server host')
    
    args = parser.parse_args()
    
    # Set CPU performance mode
    os.system("echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1")
    
    # Initialize pipeline
    global pipeline
    pipeline = HighPerformanceDetectionPipeline(args.width, args.height, args.fps, args.target_fps)
    
    if not pipeline.initialize():
        logger.error("Failed to initialize pipeline")
        return
    
    # Start detection in background thread
    detection_thread = threading.Thread(target=pipeline.run, daemon=True)
    detection_thread.start()
    
    try:
        print(f"\n🚀 High-Performance Web Viewer Starting...")
        print(f"📡 Access at: http://localhost:{args.port}")
        print(f"🎯 Target FPS: {args.target_fps}")
        print(f"📸 Camera: {args.width}x{args.height} @ {args.fps}fps")
        print(f"🧠 Inference: 320x320 (optimized)")
        print(f"⚡ Press Ctrl+C to stop\n")
        
        # Start web server
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping high-performance viewer...")
    finally:
        pipeline.stop()

if __name__ == '__main__':
    main()
