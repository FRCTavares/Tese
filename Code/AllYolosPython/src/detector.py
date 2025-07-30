"""
Object Detector using YOLOv8
Handles model loading, inference, and result processing with backend abstraction.
"""
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
import time
import logging
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

logger = logging.getLogger(__name__)


class Detection:
    """Represents a single object detection result."""
    
    def __init__(
        self, 
        bbox: Tuple[int, int, int, int], 
        confidence: float, 
        class_id: int, 
        class_name: str
    ) -> None:
        """
        Initialize detection result.
        
        Args:
            bbox: Bounding box as (x1, y1, x2, y2)
            confidence: Detection confidence score [0-1]
            class_id: Class ID from model
            class_name: Human-readable class name
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
    
    def get_center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def get_area(self) -> int:
        """Get area of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class Detector:
    """Object detector using YOLOv8 with backend abstraction for Pi 5 optimization."""
    
    def __init__(
        self, 
        model_path: str = "yolov8n.pt", 
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.4,
        backend: str = "ultralytics"
    ) -> None:
        """
        Initialize object detector.
        
        Args:
            model_path: Path to model file (default YOLOv8-nano)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            backend: Inference backend ("ultralytics", "onnx", "ncnn")
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.backend = backend
        
        self.model: Optional[Any] = None
        self.class_names: List[str] = []
        self.input_size: Tuple[int, int] = (640, 640)  # YOLOv8 default
        self.is_loaded = False
        
        # Performance tracking
        self.inference_times: List[float] = []
        self.total_detections = 0
        
    def load_model(self, model_path=None, model_name=None) -> bool:
        """
        Load YOLO model with flexible model selection.
        
        Args:
            model_path: Direct path to model file
            model_name: Model name to search for in models directory
        """
        try:
            if model_path:
                if Path(model_path).exists():
                    self.model = YOLO(model_path)
                    self.model_loaded = True
                    print(f"✅ Loaded model: {model_path}")
                    return True
                else:
                    print(f"❌ Model not found: {model_path}")
            
            if model_name:
                # Search for model in models directory
                models_dir = Path("models")
                if models_dir.exists():
                    # Try exact match first
                    exact_match = models_dir / model_name
                    if exact_match.exists():
                        self.model = YOLO(str(exact_match))
                        self.model_loaded = True
                        print(f"✅ Loaded model: {exact_match}")
                        return True
                    
                    # Try pattern matching
                    matches = list(models_dir.glob(f"*{model_name}*"))
                    if matches:
                        model_file = matches[0]  # Use first match
                        self.model = YOLO(str(model_file))
                        self.model_loaded = True
                        print(f"✅ Loaded model: {model_file}")
                        return True
                    
                    print(f"❌ No model found matching: {model_name}")
            
            # Default fallback
            default_models = ["models/yolov8n.pt", "yolov8n.pt"]
            for default_model in default_models:
                try:
                    self.model = YOLO(default_model)
                    self.model_loaded = True
                    print(f"✅ Loaded default model: {default_model}")
                    return True
                except Exception:
                    continue
                    
            print("❌ No models available. Run 'python scripts/download_models.py' to download models.")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.model_loaded = False
        
        return False
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run object detection on an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of Detection objects
        """
        if not self.is_loaded or not self.model:
            logger.warning("Model not loaded")
            return []
        
        start_time = time.time()
        
        try:
            if self.backend == "ultralytics":
                # Run inference
                results = self.model(
                    image, 
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                detections = []
                if results and len(results) > 0:
                    result = results[0]  # First result
                    
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        
                        for i in range(len(boxes)):
                            # Get bounding box coordinates
                            box = boxes.xyxy[i].cpu().numpy()
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Get confidence and class
                            confidence = float(boxes.conf[i].cpu().numpy())
                            class_id = int(boxes.cls[i].cpu().numpy())
                            
                            # Get class name
                            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                            
                            detection = Detection((x1, y1, x2, y2), confidence, class_id, class_name)
                            detections.append(detection)
                
                # Track performance
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                self.total_detections += len(detections)
                
                # Keep only last 100 inference times for moving average
                if len(self.inference_times) > 100:
                    self.inference_times.pop(0)
                
                return detections
                
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
        
        return []
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {'avg_inference_time': 0.0, 'fps': 0.0, 'total_detections': 0}
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0.0
        
        return {
            'avg_inference_time': avg_time,
            'fps': fps,
            'total_detections': self.total_detections,
            'recent_samples': len(self.inference_times)
        }
    
    def annotate_frame(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Annotate image with detection results.
        
        Args:
            image: Input image
            detections: List of detections to draw
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            confidence = detection.confidence
            class_name = detection.class_name
            
            # Choose color based on class
            color = self._get_class_color(detection.class_id)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background rectangle for text
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return annotated
    
    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get consistent color for a class ID."""
        # Generate consistent colors based on class ID
        np.random.seed(class_id)
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        return color
