#!/usr/bin/env python3
"""
YOLO Object Detection Node for UAV Applications
Subscribes to camera images and publishes object detections.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

# Import custom messages (will be generated)
from uav_object_detection.msg import Detection, DetectionArray

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available. Using mock detector.")


class DetectorNode(Node):
    """ROS2 node for YOLO-based object detection."""
    
    def __init__(self):
        super().__init__('detector_node')
        
        # Declare parameters
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('nms_threshold', 0.4)
        self.declare_parameter('input_size', 640)
        self.declare_parameter('device', 'cpu')  # 'cpu', 'cuda', 'mps'
        self.declare_parameter('max_detections', 100)
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.nms_threshold = self.get_parameter('nms_threshold').get_parameter_value().double_value
        self.input_size = self.get_parameter('input_size').get_parameter_value().integer_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.max_detections = self.get_parameter('max_detections').get_parameter_value().integer_value
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # QoS profile for real-time processing
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # Subscriber
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            image_qos
        )
        
        # Publishers
        self.detection_pub = self.create_publisher(
            DetectionArray,
            '/detections',
            10
        )
        
        self.annotated_image_pub = self.create_publisher(
            Image,
            '/detections/annotated_image',
            image_qos
        )
        
        # Initialize detector
        self.detector = None
        if ULTRALYTICS_AVAILABLE:
            self.detector = YOLODetector(
                self.model_path,
                self.confidence_threshold,
                self.nms_threshold,
                self.input_size,
                self.device
            )
        else:
            self.detector = MockDetector()
        
        if not self.detector.load_model():
            self.get_logger().error("Failed to load detection model")
            return
        
        self.get_logger().info(f"Detector node started with model: {self.model_path}")
        self.get_logger().info(f"Confidence threshold: {self.confidence_threshold}")
        self.get_logger().info(f"Device: {self.device}")
        
        # Performance tracking
        self.detection_count = 0
        self.total_inference_time = 0.0
        self.last_stats_time = time.time()
        
    def image_callback(self, msg: Image):
        """Process incoming images and publish detections."""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Run detection
            start_time = time.time()
            detections = self.detector.detect(cv_image)
            inference_time_ms = (time.time() - start_time) * 1000.0
            
            # Create detection message
            detection_array = DetectionArray()
            detection_array.header = msg.header
            detection_array.total_detections = len(detections)
            detection_array.inference_time_ms = inference_time_ms
            detection_array.model_name = self.model_path
            
            # Convert detections to ROS messages
            for det in detections:
                detection_msg = Detection()
                detection_msg.header = msg.header
                detection_msg.class_id = det['class_id']
                detection_msg.class_name = det['class_name']
                detection_msg.confidence = det['confidence']
                
                # Bounding box
                x1, y1, x2, y2 = det['bbox']
                detection_msg.bbox_min.x = float(x1)
                detection_msg.bbox_min.y = float(y1)
                detection_msg.bbox_min.z = 0.0
                detection_msg.bbox_max.x = float(x2)
                detection_msg.bbox_max.y = float(y2)
                detection_msg.bbox_max.z = 0.0
                
                # Center point
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                detection_msg.center.x = center_x
                detection_msg.center.y = center_y
                detection_msg.center.z = 0.0
                
                # Area
                detection_msg.area = float((x2 - x1) * (y2 - y1))
                
                detection_array.detections.append(detection_msg)
            
            # Publish detections
            self.detection_pub.publish(detection_array)
            
            # Create and publish annotated image
            annotated_image = self.draw_detections(cv_image, detections)
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            annotated_msg.header = msg.header
            self.annotated_image_pub.publish(annotated_msg)
            
            # Update statistics
            self.detection_count += 1
            self.total_inference_time += inference_time_ms
            
            current_time = time.time()
            if current_time - self.last_stats_time >= 10.0:  # Log stats every 10 seconds
                avg_inference_time = self.total_inference_time / self.detection_count
                detection_rate = self.detection_count / (current_time - self.last_stats_time)
                
                self.get_logger().info(
                    f"Detection stats - Rate: {detection_rate:.1f} Hz, "
                    f"Avg inference: {avg_inference_time:.1f} ms, "
                    f"Total detections in frame: {len(detections)}"
                )
                
                # Reset counters
                self.detection_count = 0
                self.total_inference_time = 0.0
                self.last_stats_time = current_time
            
        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Draw bounding box
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for label
            cv2.rectangle(
                annotated,
                (int(x1), int(y1) - label_size[1] - 10),
                (int(x1) + label_size[0], int(y1)),
                (0, 255, 0),
                -1
            )
            
            # Label text
            cv2.putText(
                annotated,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )
        
        return annotated


class YOLODetector:
    """YOLO detection wrapper."""
    
    def __init__(self, model_path: str, confidence_threshold: float, 
                 nms_threshold: float, input_size: int, device: str):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.device = device
        self.model = None
        
    def load_model(self) -> bool:
        """Load YOLO model."""
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            return True
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            return False
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Run object detection on image."""
        if self.model is None:
            return []
        
        try:
            # Run inference
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                imgsz=self.input_size,
                verbose=False
            )
            
            detections = []
            
            # Process results
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i in range(len(boxes)):
                        detection = {
                            'bbox': boxes[i].tolist(),
                            'confidence': float(confidences[i]),
                            'class_id': int(class_ids[i]),
                            'class_name': self.model.names[class_ids[i]]
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []


class MockDetector:
    """Mock detector for testing without YOLO."""
    
    def __init__(self):
        self.frame_count = 0
        
    def load_model(self) -> bool:
        """Mock model loading."""
        return True
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Generate mock detections."""
        height, width = image.shape[:2]
        detections = []
        
        # Create some fake detections
        if self.frame_count % 30 == 0:  # Detection every 30 frames
            detections.append({
                'bbox': [width*0.2, height*0.2, width*0.4, height*0.4],
                'confidence': 0.85,
                'class_id': 0,
                'class_name': 'person'
            })
            
        if self.frame_count % 45 == 0:  # Another detection
            detections.append({
                'bbox': [width*0.6, height*0.1, width*0.9, height*0.3],
                'confidence': 0.72,
                'class_id': 2,
                'class_name': 'car'
            })
        
        self.frame_count += 1
        return detections


def main(args=None):
    rclpy.init(args=args)
    
    detector_node = DetectorNode()
    
    try:
        rclpy.spin(detector_node)
    except KeyboardInterrupt:
        pass
    finally:
        detector_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
