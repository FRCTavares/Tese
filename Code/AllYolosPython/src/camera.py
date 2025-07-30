"""
RealSense D435 Camera Streamer
Handles Intel RealSense D435 camera initialization and frame capture.
"""
from typing import Optional, Tuple
import pyrealsense2 as rs
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


class RealSenseStreamer:
    """Manages Intel RealSense D435 camera streaming with optimized settings for Pi 5."""
    
    def __init__(
        self, 
        width: int = 480, 
        height: int = 270, 
        fps: int = 30,
        enable_depth: bool = False
    ) -> None:
        """
        Initialize RealSense camera streamer.
        
        Args:
            width: Frame width in pixels (default 480 for Pi 5 optimization)
            height: Frame height in pixels (default 270 for Pi 5 optimization) 
            fps: Frames per second (default 30)
            enable_depth: Whether to enable depth stream (default False for performance)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        
        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None
        self.align: Optional[rs.align] = None
        self.is_streaming = False
        
    def start(self) -> bool:
        """
        Start the camera streaming.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize pipeline and config
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Configure RGB stream - optimized resolution for Pi 5
            self.config.enable_stream(
                rs.stream.color, 
                self.width, 
                self.height, 
                rs.format.bgr8, 
                self.fps
            )
            
            # Optionally enable depth stream
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
            profile = self.pipeline.start(self.config)
            
            # Get device and configure settings for optimal performance
            device = profile.get_device()
            color_sensor = device.first_color_sensor()
            
            # Disable auto-exposure and set manual exposure for consistent performance
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, True)
            
            # Set auto white balance
            if color_sensor.supports(rs.option.enable_auto_white_balance):
                color_sensor.set_option(rs.option.enable_auto_white_balance, True)
            
            self.is_streaming = True
            logger.info(f"RealSense camera started: {self.width}x{self.height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start RealSense camera: {e}")
            return False
    
    def get_frame(self) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Capture a frame from the camera.
        
        Returns:
            Tuple of (color_frame, depth_frame) or None if failed
            depth_frame is None if depth is not enabled
        """
        if not self.is_streaming or not self.pipeline:
            return None
            
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            
            if self.enable_depth and self.align:
                # Align depth to color
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
            else:
                color_frame = frames.get_color_frame()
                depth_frame = None
            
            if not color_frame:
                return None
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
            
            return color_image, depth_image
            
        except Exception as e:
            logger.warning(f"Failed to capture frame: {e}")
            return None
    
    def stop(self) -> None:
        """Stop the camera streaming."""
        if self.pipeline and self.is_streaming:
            try:
                self.pipeline.stop()
                self.is_streaming = False
                logger.info("RealSense camera stopped")
            except Exception as e:
                logger.error(f"Error stopping camera: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def get_camera_info(self) -> dict:
        """
        Get camera information and capabilities.
        
        Returns:
            Dictionary with camera information
        """
        if not self.pipeline:
            return {}
        
        try:
            profile = self.pipeline.get_active_profile()
            device = profile.get_device()
            
            info = {
                'name': device.get_info(rs.camera_info.name),
                'serial_number': device.get_info(rs.camera_info.serial_number),
                'firmware_version': device.get_info(rs.camera_info.firmware_version),
                'product_id': device.get_info(rs.camera_info.product_id),
                'usb_type': device.get_info(rs.camera_info.usb_type_descriptor),
                'width': self.width,
                'height': self.height,
                'fps': self.fps,
                'depth_enabled': self.enable_depth
            }
            return info
        except Exception as e:
            logger.error(f"Failed to get camera info: {e}")
            return {}
