#!/usr/bin/env python3
"""
RealSense Camera Node for UAV Object Detection
Publishes color and depth images from Intel RealSense D435 camera.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from typing import Optional, Tuple

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: pyrealsense2 not available. Using mock camera.")


class CameraNode(Node):
    """ROS2 node for RealSense camera streaming."""
    
    def __init__(self):
        super().__init__('camera_node')
        
        # Declare parameters
        self.declare_parameter('width', 480)
        self.declare_parameter('height', 270)
        self.declare_parameter('fps', 30)
        self.declare_parameter('enable_depth', False)
        self.declare_parameter('publish_rate', 30.0)
        
        # Get parameters
        self.width = self.get_parameter('width').get_parameter_value().integer_value
        self.height = self.get_parameter('height').get_parameter_value().integer_value
        self.fps = self.get_parameter('fps').get_parameter_value().integer_value
        self.enable_depth = self.get_parameter('enable_depth').get_parameter_value().bool_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # QoS profile for real-time image streaming
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # Publishers
        self.color_pub = self.create_publisher(
            Image, 
            '/camera/color/image_raw', 
            image_qos
        )
        
        self.color_info_pub = self.create_publisher(
            CameraInfo,
            '/camera/color/camera_info',
            image_qos
        )
        
        if self.enable_depth:
            self.depth_pub = self.create_publisher(
                Image,
                '/camera/depth/image_raw',
                image_qos
            )
            
            self.depth_info_pub = self.create_publisher(
                CameraInfo,
                '/camera/depth/camera_info',
                image_qos
            )
        
        # Initialize camera
        self.camera = None
        if REALSENSE_AVAILABLE:
            self.camera = RealSenseCamera(
                self.width, 
                self.height, 
                self.fps, 
                self.enable_depth
            )
        else:
            self.camera = MockCamera(
                self.width, 
                self.height, 
                self.fps
            )
        
        # Start camera
        if not self.camera.start():
            self.get_logger().error("Failed to start camera")
            return
            
        self.get_logger().info(f"Camera node started: {self.width}x{self.height}@{self.fps}fps")
        
        # Create timer for publishing
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # Frame counter for diagnostics
        self.frame_count = 0
        self.last_fps_time = time.time()
        
    def timer_callback(self):
        """Timer callback to capture and publish frames."""
        try:
            # Get frames from camera
            color_frame, depth_frame = self.camera.get_frames()
            
            if color_frame is None:
                return
            
            # Create header with current timestamp
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "camera_color_optical_frame"
            
            # Publish color image
            color_msg = self.bridge.cv2_to_imgmsg(color_frame, encoding="bgr8")
            color_msg.header = header
            self.color_pub.publish(color_msg)
            
            # Publish camera info
            color_info = self.create_camera_info(header, "color")
            self.color_info_pub.publish(color_info)
            
            # Publish depth image if enabled
            if self.enable_depth and depth_frame is not None:
                depth_header = Header()
                depth_header.stamp = header.stamp
                depth_header.frame_id = "camera_depth_optical_frame"
                
                depth_msg = self.bridge.cv2_to_imgmsg(depth_frame, encoding="16UC1")
                depth_msg.header = depth_header
                self.depth_pub.publish(depth_msg)
                
                depth_info = self.create_camera_info(depth_header, "depth")
                self.depth_info_pub.publish(depth_info)
            
            # Update diagnostics
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 5.0:  # Log FPS every 5 seconds
                fps = self.frame_count / (current_time - self.last_fps_time)
                self.get_logger().info(f"Camera FPS: {fps:.1f}")
                self.frame_count = 0
                self.last_fps_time = current_time
                
        except Exception as e:
            self.get_logger().error(f"Error in timer callback: {e}")
    
    def create_camera_info(self, header: Header, camera_type: str) -> CameraInfo:
        """Create camera info message."""
        info = CameraInfo()
        info.header = header
        info.width = self.width
        info.height = self.height
        
        # Basic camera parameters (would need calibration for real use)
        if camera_type == "color":
            # Typical values for RealSense D435 color camera
            fx = fy = 400.0  # Focal length
            cx = self.width / 2.0
            cy = self.height / 2.0
        else:  # depth
            fx = fy = 400.0
            cx = self.width / 2.0
            cy = self.height / 2.0
        
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # No distortion
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        
        return info
    
    def destroy_node(self):
        """Clean up when node is destroyed."""
        if self.camera:
            self.camera.stop()
        super().destroy_node()


class RealSenseCamera:
    """RealSense camera interface."""
    
    def __init__(self, width: int, height: int, fps: int, enable_depth: bool):
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        
        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None
        self.align: Optional[rs.align] = None
        self.is_streaming = False
        
    def start(self) -> bool:
        """Start camera streaming."""
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Configure color stream
            self.config.enable_stream(
                rs.stream.color, 
                self.width, 
                self.height, 
                rs.format.bgr8, 
                self.fps
            )
            
            # Configure depth stream if enabled
            if self.enable_depth:
                self.config.enable_stream(
                    rs.stream.depth, 
                    self.width, 
                    self.height, 
                    rs.format.z16, 
                    self.fps
                )
                self.align = rs.align(rs.stream.color)
            
            # Start streaming
            self.pipeline.start(self.config)
            self.is_streaming = True
            
            return True
            
        except Exception as e:
            print(f"Failed to start RealSense camera: {e}")
            return False
    
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get color and depth frames."""
        if not self.is_streaming:
            return None, None
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            
            # Get color frame
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None, None
            
            color_image = np.asanyarray(color_frame.get_data())
            
            # Get depth frame if enabled
            depth_image = None
            if self.enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    # Align depth to color
                    if self.align:
                        aligned_frames = self.align.process(frames)
                        depth_frame = aligned_frames.get_depth_frame()
                    depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_image
            
        except Exception as e:
            print(f"Error getting frames: {e}")
            return None, None
    
    def stop(self):
        """Stop camera streaming."""
        if self.is_streaming and self.pipeline:
            self.pipeline.stop()
            self.is_streaming = False


class MockCamera:
    """Mock camera for testing without RealSense hardware."""
    
    def __init__(self, width: int, height: int, fps: int):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        
    def start(self) -> bool:
        """Start mock camera."""
        return True
    
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Generate mock frames."""
        # Create a test pattern
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add some visual elements
        cv2.rectangle(frame, (50, 50), (150, 150), (0, 255, 0), 2)
        cv2.circle(frame, (self.width//2, self.height//2), 30, (255, 0, 0), -1)
        cv2.putText(frame, f"Frame {self.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        self.frame_count += 1
        
        return frame, None
    
    def stop(self):
        """Stop mock camera."""
        pass


def main(args=None):
    rclpy.init(args=args)
    
    camera_node = CameraNode()
    
    try:
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        pass
    finally:
        camera_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
