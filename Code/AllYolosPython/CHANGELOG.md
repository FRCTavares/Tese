# Changelog

All notable changes to the Real-Time Object Detection Pipeline project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-07-29

### Added
- Initial release of Real-Time Object Detection Pipeline for Raspberry Pi 5
- Intel RealSense D435 camera integration with optimized streaming
- YOLOv8-nano object detection with 80 COCO classes
- Multiple viewing modes: GUI, headless, web-based, high-performance
- Comprehensive setup scripts for system configuration and RealSense SDK
- Performance optimization pipeline achieving 3.5+ FPS on Pi 5
- Web-based live viewer with Flask server and browser streaming
- High-performance mode with dual-resolution processing and frame skipping
- Mock camera support for testing without hardware
- Complete documentation suite with installation, optimization, and troubleshooting guides
- Automated system health checks and performance benchmarking
- Flexible YAML-based configuration system
- Professional project structure with proper organization

### Features
- **Camera Support**: Intel RealSense D435 with multiple resolution options
- **Detection**: YOLOv8-nano with configurable confidence thresholds
- **Performance**: 1.9-3.5+ FPS depending on mode and settings
- **Viewing Options**: 
  - GUI mode with OpenCV display
  - Web viewer accessible via browser (http://localhost:5000)
  - High-performance mode with optimizations
  - Headless mode for background processing
- **Configurations**: Default, high-performance, and quality presets
- **Platform**: Optimized for Raspberry Pi 5 (8GB RAM) with Ubuntu 24.04

### Technical Details
- **Languages**: Python 3.11+
- **Frameworks**: Ultralytics YOLO, OpenCV, Flask, pyrealsense2
- **Architecture**: Modular design with camera, detector, and viewer components
- **Optimization**: Frame skipping, dual-resolution processing, confidence filtering
- **Testing**: Comprehensive test suite with FPS validation and benchmarking

### Documentation
- Complete installation guide with step-by-step instructions
- Performance optimization strategies and system tuning
- Troubleshooting guide for common issues
- Development log with full project timeline
- API documentation for custom development

### Scripts and Tools
- Automated system setup and RealSense SDK installation
- Virtual environment configuration and library path fixes
- System health checks and diagnostic tools
- Performance benchmarking and monitoring
- Code formatting and quality checks

### Hardware Requirements
- Raspberry Pi 5 (8GB RAM recommended)
- Intel RealSense D435 camera
- USB 3.0 connection
- 5V/5A power supply
- Ubuntu 24.04 LTS

### Performance Benchmarks
- **Standard Mode**: ~1.9 FPS (424×240)
- **Web Viewer**: ~2.5 FPS (640×480 streaming)  
- **High-Performance**: ~3.5+ FPS (optimized pipeline)
- **Mock Camera**: 25+ FPS (synthetic frames)

### Known Issues
- RealSense SDK requires manual compilation on Pi 5 (30-60 minutes)
- Custom camera resolutions may cause "Couldn't resolve requests" errors
- Performance varies with system temperature and power supply quality
- Web viewer requires manual browser access to localhost:5000

### Future Roadmap
- ONNX backend support for faster inference
- NCNN int8 quantization for mobile optimization
- Model pruning and custom training capabilities
- TensorRT support for NVIDIA platforms
- Docker containerization
- API documentation completion

## [Unreleased]

### Planned
- [ ] ONNX export and inference backend
- [ ] NCNN int8 quantization support
- [ ] Model pruning utilities
- [ ] Docker containerization
- [ ] API reference documentation
- [ ] Custom model training examples
- [ ] Edge TPU support investigation
- [ ] Performance optimization for 5+ FPS target
