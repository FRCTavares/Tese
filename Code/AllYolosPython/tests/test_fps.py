"""
Tests for Real-Time Object Detection Pipeline
Validates FPS performance and basic functionality.
"""
import pytest
import numpy as np
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from detector import Detector, Detection
from camera import RealSenseStreamer


class TestDetector:
    """Test cases for the Detector class."""
    
    def test_detector_initialization(self):
        """Test detector can be initialized."""
        detector = Detector(
            model_path="yolov8n.pt",
            confidence_threshold=0.5,
            backend="ultralytics"
        )
        assert detector.model_path == "yolov8n.pt"
        assert detector.confidence_threshold == 0.5
        assert detector.backend == "ultralytics"
        assert not detector.is_loaded
    
    def test_detection_class(self):
        """Test Detection result class."""
        detection = Detection(
            bbox=(10, 20, 100, 200),
            confidence=0.85,
            class_id=1,
            class_name="person"
        )
        
        assert detection.bbox == (10, 20, 100, 200)
        assert detection.confidence == 0.85
        assert detection.class_id == 1
        assert detection.class_name == "person"
        
        # Test utility methods
        center = detection.get_center()
        assert center == (55, 110)  # ((10+100)/2, (20+200)/2)
        
        area = detection.get_area()
        assert area == 16200  # (100-10) * (200-20)


class TestRealSenseStreamer:
    """Test cases for the RealSenseStreamer class."""
    
    def test_streamer_initialization(self):
        """Test camera streamer can be initialized."""
        streamer = RealSenseStreamer(
            width=480,
            height=270,
            fps=30,
            enable_depth=False
        )
        
        assert streamer.width == 480
        assert streamer.height == 270
        assert streamer.fps == 30
        assert not streamer.enable_depth
        assert not streamer.is_streaming


class TestPipelinePerformance:
    """Test cases for pipeline performance validation."""
    
    @pytest.mark.slow
    @patch('src.detector.YOLO')
    def test_detector_fps_mock(self, mock_yolo):
        """Test that detector returns positive FPS with mocked inference."""
        # Mock YOLO model
        mock_model = Mock()
        mock_result = Mock()
        mock_boxes = Mock()
        
        # Configure mocks
        mock_boxes.xyxy = [np.array([10, 20, 100, 200])]
        mock_boxes.conf = [np.array([0.85])]
        mock_boxes.cls = [np.array([0])]
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        mock_model.names = {0: 'person'}
        mock_yolo.return_value = mock_model
        
        # Create detector and load model
        detector = Detector(model_path="yolov8n.pt", backend="ultralytics")
        detector.load_model()
        
        # Run multiple inferences to measure FPS
        test_image = np.zeros((270, 480, 3), dtype=np.uint8)
        num_runs = 10
        
        start_time = time.time()
        for _ in range(num_runs):
            detections = detector.detect(test_image)
            assert len(detections) == 1  # Should detect one mocked object
        
        total_time = time.time() - start_time
        fps = num_runs / total_time
        
        # Assert positive FPS
        assert fps > 0, f"FPS should be positive, got {fps}"
        
        # Get performance stats
        stats = detector.get_performance_stats()
        assert stats['fps'] > 0, "Detector should report positive FPS"
        assert stats['total_detections'] == num_runs, f"Should have {num_runs} total detections"
        
        print(f"\nMocked inference performance: {fps:.2f} FPS")
        print(f"Average inference time: {stats['avg_inference_time']:.4f}s")
    
    def test_fps_calculation_accuracy(self):
        """Test FPS calculation accuracy with known timing."""
        detector = Detector()
        
        # Simulate known inference times
        inference_times = [0.1, 0.1, 0.1, 0.1, 0.1]  # 100ms each = 10 FPS
        detector.inference_times = inference_times
        
        stats = detector.get_performance_stats()
        expected_fps = 10.0  # 1 / 0.1
        
        assert abs(stats['fps'] - expected_fps) < 0.01, f"Expected ~{expected_fps} FPS, got {stats['fps']}"
        assert stats['avg_inference_time'] == 0.1, f"Expected 0.1s avg time, got {stats['avg_inference_time']}"
    
    @pytest.mark.benchmark
    def test_numpy_operations_performance(self):
        """Benchmark numpy operations used in the pipeline."""
        # Test image processing operations performance
        image = np.random.randint(0, 255, (270, 480, 3), dtype=np.uint8)
        
        start_time = time.time()
        for _ in range(100):
            # Simulate common image operations
            resized = image[::2, ::2]  # Downsampling
            normalized = image.astype(np.float32) / 255.0  # Normalization
            result = normalized * 255  # Scaling
        
        total_time = time.time() - start_time
        ops_per_second = 100 / total_time
        
        print(f"\nNumPy operations performance: {ops_per_second:.2f} ops/sec")
        
        # Should be able to do at least 30 operations per second on Pi 5
        assert ops_per_second >= 30, f"NumPy operations too slow: {ops_per_second} ops/sec"


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_mock_pipeline(self):
        """Test end-to-end pipeline with mocked camera and detector."""
        # This test validates the pipeline structure without hardware dependencies
        
        # Mock camera frame
        mock_frame = np.random.randint(0, 255, (270, 480, 3), dtype=np.uint8)
        
        # Mock detection
        mock_detection = Detection(
            bbox=(50, 50, 150, 150),
            confidence=0.9,
            class_id=0,
            class_name="person"
        )
        
        # Test detection annotation
        detector = Detector()
        annotated = detector.annotate_frame(mock_frame, [mock_detection])
        
        # Verify frame dimensions unchanged
        assert annotated.shape == mock_frame.shape
        
        # Verify annotation was applied (frame should be different)
        assert not np.array_equal(annotated, mock_frame), "Frame should be annotated"


def test_project_structure():
    """Test that required project files exist."""
    project_root = Path(__file__).parent.parent
    
    # Check required directories
    required_dirs = ['src', 'models', 'scripts', 'docs', 'notebooks', 'tests']
    for dir_name in required_dirs:
        assert (project_root / dir_name).exists(), f"Missing directory: {dir_name}"
    
    # Check required files
    required_files = [
        'requirements.txt',
        'setup.sh',
        'src/main.py',
        'src/camera.py',
        'src/detector.py'
    ]
    for file_path in required_files:
        assert (project_root / file_path).exists(), f"Missing file: {file_path}"


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_fps.py -v
    pytest.main([__file__, "-v", "--tb=short"])
