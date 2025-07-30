"""
Mock Camera Module for Testing Without RealSense D435
Simulates the RealSense D435 camera for testing the detection pipeline.
"""
from typing import Optional, Tuple
import numpy as np
import cv2
import time
import logging

logger = logging.getLogger(__name__)


class MockRealSenseStreamer:
    """Mock RealSense camera for testing without hardware."""
    
    def __init__(
        self, 
        width: int = 480, 
        height: int = 270, 
        fps: int = 30,
        enable_depth: bool = False
    ) -> None:
        """Initialize mock camera with test patterns."""
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        self.is_streaming = False
        self.frame_count = 0
        
        # Create test patterns
        self._create_test_patterns()
        
    def _create_test_patterns(self) -> None:
        """Create various test patterns for simulation."""
        self.test_patterns = []
        
        # Pattern 1: Gradient with moving rectangle (simulates person)
        pattern1 = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        pattern1[:, :, 0] = np.linspace(0, 255, self.width)[None, :]  # Red gradient
        pattern1[:, :, 1] = np.linspace(0, 255, self.height)[:, None]  # Green gradient
        pattern1[:, :, 2] = 100  # Blue constant
        self.test_patterns.append(pattern1)
        
        # Pattern 2: Checkerboard
        pattern2 = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for i in range(0, self.height, 40):
            for j in range(0, self.width, 40):
                if (i//40 + j//40) % 2 == 0:
                    pattern2[i:i+40, j:j+40] = [255, 255, 255]
        self.test_patterns.append(pattern2)
        
        # Pattern 3: Random noise with colored rectangles
        pattern3 = np.random.randint(50, 150, (self.height, self.width, 3), dtype=np.uint8)
        # Add some colored rectangles (simulate objects)
        cv2.rectangle(pattern3, (50, 50), (150, 150), (255, 0, 0), -1)  # Red rectangle
        cv2.rectangle(pattern3, (200, 100), (350, 200), (0, 255, 0), -1)  # Green rectangle
        cv2.rectangle(pattern3, (100, 180), (200, 250), (0, 0, 255), -1)  # Blue rectangle
        self.test_patterns.append(pattern3)
        
    def start(self) -> bool:
        """Start the mock camera streaming."""
        logger.info(f"Mock RealSense camera started: {self.width}x{self.height} @ {self.fps}fps")
        self.is_streaming = True
        return True
    
    def get_frame(self) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Get a mock frame with animated content.
        
        Returns:
            Tuple of (color_frame, depth_frame) or None if failed
        """
        if not self.is_streaming:
            return None
        
        # Cycle through test patterns
        pattern_idx = (self.frame_count // 30) % len(self.test_patterns)
        base_frame = self.test_patterns[pattern_idx].copy()
        
        # Add animated moving rectangle to simulate detected objects
        x = int(50 + 100 * np.sin(self.frame_count * 0.1))
        y = int(50 + 50 * np.cos(self.frame_count * 0.1))
        cv2.rectangle(base_frame, (x, y), (x+80, y+80), (255, 255, 0), 3)
        
        # Add frame counter text
        cv2.putText(
            base_frame, 
            f"Mock Frame {self.frame_count}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        self.frame_count += 1
        
        # Simulate camera FPS
        time.sleep(1.0 / self.fps)
        
        depth_frame = None
        if self.enable_depth:
            # Create mock depth frame
            depth_frame = np.random.randint(500, 3000, (self.height, self.width), dtype=np.uint16)
        
        return base_frame, depth_frame
    
    def stop(self) -> None:
        """Stop the mock camera streaming."""
        if self.is_streaming:
            self.is_streaming = False
            logger.info("Mock RealSense camera stopped")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def get_camera_info(self) -> dict:
        """Get mock camera information."""
        return {
            'name': 'Mock Intel RealSense D435',
            'serial_number': 'MOCK_SERIAL_123456',
            'firmware_version': '5.12.7.100',
            'product_id': '0x0B07',
            'usb_type': 'USB 3.2',
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'depth_enabled': self.enable_depth
        }
