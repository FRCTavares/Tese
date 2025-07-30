#!/usr/bin/env python3
"""
Visualization Node for UAV Object Detection
Displays detection results and system performance metrics.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import Optional

# Import custom messages
from uav_object_detection.msg import DetectionArray


class VisualizationNode(Node):
    """ROS2 node for visualizing detection results."""
    
    def __init__(self):
        super().__init__('visualization_node')
        
        # Declare parameters
        self.declare_parameter('display_window', True)
        self.declare_parameter('save_video', False)
        self.declare_parameter('video_output_path', '/tmp/detections.mp4')
        self.declare_parameter('fps', 30)
        
        # Get parameters
        self.display_window = self.get_parameter('display_window').get_parameter_value().bool_value
        self.save_video = self.get_parameter('save_video').get_parameter_value().bool_value
        self.video_output_path = self.get_parameter('video_output_path').get_parameter_value().string_value
        self.fps = self.get_parameter('fps').get_parameter_value().integer_value
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # QoS profile
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/detections/annotated_image',
            self.image_callback,
            image_qos
        )
        
        self.detection_sub = self.create_subscription(
            DetectionArray,
            '/detections',
            self.detection_callback,
            10
        )
        
        # Video writer
        self.video_writer: Optional[cv2.VideoWriter] = None
        
        # State
        self.latest_detections: Optional[DetectionArray] = None
        self.frame_count = 0
        
        self.get_logger().info("Visualization node started")
        if self.display_window:
            self.get_logger().info("Display window enabled")
        if self.save_video:
            self.get_logger().info(f"Video saving enabled: {self.video_output_path}")
    
    def detection_callback(self, msg: DetectionArray):
        """Store latest detection information."""
        self.latest_detections = msg
    
    def image_callback(self, msg: Image):
        """Display annotated image and save video if enabled."""
        try:
            # Convert to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Add performance overlay
            if self.latest_detections:
                cv_image = self.add_performance_overlay(cv_image, self.latest_detections)
            
            # Initialize video writer if needed
            if self.save_video and self.video_writer is None:
                height, width = cv_image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    self.video_output_path, fourcc, self.fps, (width, height)
                )
            
            # Save frame to video
            if self.save_video and self.video_writer:
                self.video_writer.write(cv_image)
            
            # Display window
            if self.display_window:
                cv2.imshow('UAV Object Detection', cv_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.get_logger().info("Quit requested")
                    rclpy.shutdown()
            
            self.frame_count += 1
            
        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")
    
    def add_performance_overlay(self, image: np.ndarray, detections: DetectionArray) -> np.ndarray:
        """Add performance metrics overlay to image."""
        overlay = image.copy()
        
        # Performance text
        lines = [
            f"Detections: {detections.total_detections}",
            f"Inference: {detections.inference_time_ms:.1f}ms",
            f"Model: {detections.model_name}",
            f"Frame: {self.frame_count}"
        ]
        
        # Background rectangle
        text_height = 25
        rect_height = len(lines) * text_height + 10
        cv2.rectangle(overlay, (10, 10), (300, rect_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (300, rect_height), (255, 255, 255), 2)
        
        # Add text
        for i, line in enumerate(lines):
            y_pos = 35 + i * text_height
            cv2.putText(
                overlay, line, (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
        
        return overlay
    
    def destroy_node(self):
        """Clean up resources."""
        if self.video_writer:
            self.video_writer.release()
        
        if self.display_window:
            cv2.destroyAllWindows()
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    viz_node = VisualizationNode()
    
    try:
        rclpy.spin(viz_node)
    except KeyboardInterrupt:
        pass
    finally:
        viz_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
