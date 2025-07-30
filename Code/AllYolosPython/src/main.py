"""
Real-Time Object Detection Pipeline - Main Entry Point
Raspberry Pi 5 optimized real-time object detection using Intel RealSense D435 camera.
"""
import argparse
import cv2
import numpy as np
import time
import logging
import sys
from pathlib import Path
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ObjectDetectionPipeline:
    """Main pipeline class that orchestrates camera and detection."""
    
    def __init__(
        self,
        camera_width: int = 480,
        camera_height: int = 270,
        camera_fps: int = 30,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        headless: bool = False,
        output_dir: str = "/tmp/output"
    ) -> None:
        """
        Initialize the detection pipeline.
        
        Args:
            camera_width: Camera frame width
            camera_height: Camera frame height  
            camera_fps: Camera FPS
            model_path: Path to YOLO model
            confidence_threshold: Detection confidence threshold
            headless: Run without GUI (save frames to disk)
            output_dir: Directory for headless output
        """
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.headless = headless
        self.output_dir = Path(output_dir)
        
        # Initialize components
        self.camera: Optional[RealSenseStreamer] = None
        self.detector: Optional[Detector] = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = 0.0
        self.fps_history = []
        
        # Create output directory for headless mode
        if self.headless:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Headless mode: saving frames to {self.output_dir}")
    
    def initialize(self) -> bool:
        """
        Initialize camera and detector.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Initializing Real-Time Object Detection Pipeline...")
        
        if not REALSENSE_AVAILABLE:
            logger.warning("Using MOCK CAMERA - Install RealSense SDK for real camera support")
        
        # Initialize camera
        self.camera = RealSenseStreamer(
            width=self.camera_width,
            height=self.camera_height,
            fps=self.camera_fps,
            enable_depth=False  # Disable depth for performance
        )
        
        if not self.camera.start():
            logger.error("Failed to initialize camera")
            return False
        
        # Print camera info
        camera_info = self.camera.get_camera_info()
        if camera_info:
            logger.info(f"Camera: {camera_info.get('name', 'Unknown')}")
            logger.info(f"Resolution: {camera_info.get('width')}x{camera_info.get('height')} @ {camera_info.get('fps')}fps")
            logger.info(f"Serial: {camera_info.get('serial_number', 'Unknown')}")
        
        # Initialize detector
        self.detector = Detector(
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            backend="ultralytics"
        )
        
        if not self.detector.load_model():
            logger.error("Failed to initialize detector")
            return False
        
        logger.info("Pipeline initialization complete")
        return True
    
    def run(self) -> None:
        """Run the main detection loop."""
        if not self.camera or not self.detector:
            logger.error("Pipeline not initialized")
            return
        
        logger.info("Starting real-time object detection...")
        logger.info("Press 'q' to quit, 's' to save frame (GUI mode)")
        
        self.start_time = time.time()
        frame_save_counter = 0
        
        try:
            while True:
                # Capture frame
                frame_data = self.camera.get_frame()
                if frame_data is None:
                    logger.warning("Failed to capture frame")
                    continue
                
                color_frame, _ = frame_data
                frame_start_time = time.time()
                
                # Run detection
                detections = self.detector.detect(color_frame)
                
                # Annotate frame
                annotated_frame = self.detector.annotate_frame(color_frame, detections)
                
                # Calculate FPS
                frame_time = time.time() - frame_start_time
                instantaneous_fps = 1.0 / frame_time if frame_time > 0 else 0.0
                self.fps_history.append(instantaneous_fps)
                
                # Keep rolling average of last 30 frames
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                self.frame_count += 1
                
                # Add FPS and detection count to frame
                info_text = f"FPS: {avg_fps:.1f} | Detections: {len(detections)} | Frame: {self.frame_count}"
                cv2.putText(
                    annotated_frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Print performance info to console
                if self.frame_count % 30 == 0:  # Every 30 frames
                    total_time = time.time() - self.start_time
                    overall_fps = self.frame_count / total_time
                    detector_stats = self.detector.get_performance_stats()
                    
                    print(f"\r[Frame {self.frame_count:04d}] "
                          f"FPS: {avg_fps:.1f} (avg: {overall_fps:.1f}) | "
                          f"Detections: {len(detections):02d} | "
                          f"Inference: {detector_stats.get('avg_inference_time', 0):.3f}s", 
                          end='', flush=True)
                
                if self.headless:
                    # Save frame to disk
                    output_path = self.output_dir / f"frame_{self.frame_count:06d}.jpg"
                    cv2.imwrite(str(output_path), annotated_frame)
                    
                    # Limit output in headless mode to avoid filling disk
                    if self.frame_count >= 1000:
                        logger.info("Reached frame limit in headless mode")
                        break
                        
                else:
                    # Display frame
                    cv2.imshow('Real-Time Object Detection - Pi 5', annotated_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Quit requested")
                        break
                    elif key == ord('s'):
                        # Save current frame
                        save_path = f"detection_frame_{frame_save_counter:03d}.jpg"
                        cv2.imwrite(save_path, annotated_frame)
                        logger.info(f"Frame saved as {save_path}")
                        frame_save_counter += 1
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
        
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up...")
        
        if self.camera:
            self.camera.stop()
        
        if not self.headless:
            cv2.destroyAllWindows()
        
        # Print final statistics
        if self.frame_count > 0:
            total_time = time.time() - self.start_time
            overall_fps = self.frame_count / total_time
            
            print(f"\n\n=== Final Statistics ===")
            print(f"Total frames processed: {self.frame_count}")
            print(f"Total runtime: {total_time:.2f}s")
            print(f"Average FPS: {overall_fps:.2f}")
            
            if self.detector:
                detector_stats = self.detector.get_performance_stats()
                print(f"Total detections: {detector_stats.get('total_detections', 0)}")
                print(f"Average inference time: {detector_stats.get('avg_inference_time', 0):.3f}s")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-Time Object Detection Pipeline for Raspberry Pi 5"
    )
    
    parser.add_argument(
        "--width", 
        type=int, 
        default=480, 
        help="Camera frame width (default: 480)"
    )
    parser.add_argument(
        "--height", 
        type=int, 
        default=270, 
        help="Camera frame height (default: 270)"
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=30, 
        help="Camera FPS (default: 30)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="yolov8n.pt", 
        help="YOLO model path or name. Examples: yolov8n.pt, yolov8s.onnx, models/yolov10n.pt (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--confidence", 
        type=float, 
        default=0.5, 
        help="Detection confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--headless", 
        action="store_true", 
        help="Run without GUI, save frames to /tmp/output"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="/tmp/output", 
        help="Output directory for headless mode (default: /tmp/output)"
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = ObjectDetectionPipeline(
        camera_width=args.width,
        camera_height=args.height,
        camera_fps=args.fps,
        model_path=args.model,
        confidence_threshold=args.confidence,
        headless=args.headless,
        output_dir=args.output_dir
    )
    
    if pipeline.initialize():
        pipeline.run()
    else:
        logger.error("Failed to initialize pipeline")
        sys.exit(1)


if __name__ == "__main__":
    main()
