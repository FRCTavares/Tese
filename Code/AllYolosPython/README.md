# Real-Time Object Detection Pipeline for Raspberry Pi 5

A lightweight, optimized real-time object detection system designed specifically for Raspberry Pi 5 (8GB RAM) with Intel RealSense D435 depth-RGB camera. Features YOLOv8-nano for efficient inference and supports GUI, headless, and web-based live viewing modes with advanced performance optimizations.

## 🚀 Quick Start

### Prerequisites
- Raspberry Pi 5 (8GB RAM recommended)
- Ubuntu 24.04 LTS
- Intel RealSense D435 camera
- USB 3.0 port for camera connection
- Python ≥ 3.11

### Installation

1. **Clone and setup:**
```bash
git clone <repository-url>
cd real-time-object-detection-pi5
chmod +x scripts/*.sh
./scripts/setup.sh
./scripts/setup_realsense.sh  # Builds RealSense SDK from source (30-60 min)
```

2. **Set environment and activate:**
```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
source venv/bin/activate
```

3. **Verify installation:**
```bash
./scripts/system_check.sh
```

3. **Run the detection pipeline:**
```bash
# GUI mode (requires monitor)
python3 src/main.py --width 640 --height 480

# Headless mode (saves frames to /tmp/output)
python3 src/main.py --headless --width 424 --height 240

# Web-based live viewer (access at http://localhost:5000)
python3 src/web_viewer.py --width 640 --height 480

# High performance mode (optimized for 5+ FPS)
python3 src/high_performance_viewer.py --width 424 --height 240
```

### Supported D435 Camera Resolutions
- **424x240 @ 30fps**: Optimal for Pi 5 performance
- **640x480 @ 30fps**: Standard VGA, good balance
- **1280x720 @ 15fps**: HD quality (may reduce FPS)

⚠️ **Important**: Always use exact supported camera resolutions. Custom resolutions like 320x240 will fail with "Couldn't resolve requests" error.

## 📊 Expected Performance

### Raspberry Pi 5 (8GB RAM) Benchmarks
- **Resolution:** 424×240 @ 30fps (optimized)
- **Achieved FPS:** 1.9-3.5 FPS (standard mode), 5+ FPS target with optimizations
- **Model:** YOLOv8-nano (6.2MB)
- **Inference time:** ~40-60ms per frame
- **Memory usage:** ~2-3GB total
- **CPU usage:** ~60-80% (4 cores utilized)

### Performance by Mode
- **Standard GUI mode:** ~1.9 FPS (424×240)
- **Web viewer mode:** ~2.5 FPS (live browser streaming)
- **High-performance mode:** ~3.5+ FPS (with optimizations)
- **Mock camera mode:** 25+ FPS (synthetic frames)

### Performance Optimization Settings
```bash
# Balanced performance (default)
python3 src/main.py --width 424 --height 240 --fps 30

# Web-based live viewer
python3 src/web_viewer.py --width 640 --height 480

# High performance mode (optimized pipeline)
python3 src/high_performance_viewer.py --width 424 --height 240

# High quality (may reduce FPS)
python3 src/main.py --width 640 --height 480 --fps 15
```

## 📁 Project Structure

```
├── src/                    # Source code
│   ├── main.py            # Main entry point (GUI/headless)
│   ├── web_viewer.py      # Web-based live viewer
│   ├── high_performance_viewer.py  # Optimized high-FPS pipeline
│   ├── camera.py          # RealSense D435 camera interface
│   ├── mock_camera.py     # Mock camera for testing
│   └── detector.py        # YOLOv8 object detection wrapper
├── docs/                  # Documentation
│   ├── installation_guide.md      # Detailed setup instructions
│   ├── performance_optimization.md # Performance tuning guide
│   ├── troubleshooting.md         # Common issues and solutions
│   └── development_log.md         # Complete development timeline
├── scripts/               # Setup and utility scripts
│   ├── setup.sh          # System setup script
│   ├── setup_realsense.sh # RealSense SDK installation
│   ├── fix_realsense_venv.sh # Fix RealSense in virtual environment
│   ├── system_check.sh   # System diagnostics
│   └── benchmark.sh      # Performance benchmarking
├── config/                # Configuration files
│   ├── default.yaml      # Default configuration
│   ├── high_performance.yaml # Performance-optimized settings
│   └── quality.yaml      # Quality-optimized settings
├── requirements/          # Dependency specifications
│   ├── requirements_pi5.txt      # Pi 5 without RealSense
│   └── requirements_no_realsense.txt # Alternative requirements
├── models/                # Model storage (auto-downloaded)
├── notebooks/             # Jupyter notebooks for training
├── tests/                 # Test suite
│   └── test_fps.py       # FPS validation tests
├── requirements.txt       # Main Python dependencies
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

## 🎯 Features

### Core Functionality
- **Real-time object detection** using YOLOv8-nano
- **80 COCO classes** (person, car, dog, etc.)
- **Live FPS monitoring** and performance statistics
- **Bounding box visualization** with confidence scores
- **Multi-backend support** (Ultralytics, ONNX, NCNN planned)

### Operation Modes
- **GUI mode:** Live preview with OpenCV window
- **Web viewer mode:** Browser-based live streaming (Flask server)
- **High-performance mode:** Optimized pipeline with aggressive optimizations
- **Headless mode:** Background processing with frame saving
- **Mock camera mode:** Testing with synthetic frames
- **Configurable resolution** and frame rates
- **Adjustable confidence thresholds**

### Pi 5 Optimizations
- **Low-resolution inference** (320×320) for speed in high-performance mode
- **Frame skipping** (adaptive rates based on processing time)
- **Fast JPEG encoding** for web streaming
- **Confidence filtering** to reduce false positives
- **Memory-efficient streaming** from RealSense D435
- **CPU-optimized inference** with automatic threading
- **Swap management** for memory stability

## 📦 Available Models

The project includes multiple YOLO models for performance comparison and optimization:

### Model Selection
| Model Family | Nano | Small | Medium | Format Support |
|-------------|------|-------|---------|----------------|
| **YOLOv8** | yolov8n.pt | yolov8s.pt | yolov8m.pt | PyTorch + ONNX |
| **YOLOv5** | yolov5n.pt | yolov5s.pt | - | PyTorch + ONNX |
| **YOLOv10** | yolov10n.pt | yolov10s.pt | - | PyTorch + ONNX |

### Performance Expectations (Raspberry Pi 5)
- **Nano models** (~5-10MB): 8-15 FPS (real-time capable)
- **Small models** (~15-25MB): 4-8 FPS (interactive)
- **Medium models** (~50MB): 2-5 FPS (high accuracy)
- **ONNX format**: Typically 10-30% faster than PyTorch

### Model Download & Comparison
```bash
# Download all models and create ONNX exports
python scripts/download_models.py

# Run comprehensive multi-model benchmark
./scripts/multi_model_benchmark.sh

# Quick comparison in Jupyter notebook
jupyter notebook notebooks/model_fine_tuning_export.ipynb
```

## ⚙️ Configuration

The pipeline supports flexible configuration through YAML files:

```bash
# Use default configuration
python3 src/main.py

# Use high-performance configuration
python3 src/main.py --config config/high_performance.yaml

# Use quality configuration  
python3 src/main.py --config config/quality.yaml

# Create custom configuration
cp config/default.yaml config/my_config.yaml
# Edit my_config.yaml as needed
python3 src/main.py --config config/my_config.yaml
```

### Configuration Profiles

- **`default.yaml`**: Balanced settings for general use
- **`high_performance.yaml`**: Optimized for maximum FPS (~3.5+ FPS)
- **`quality.yaml`**: Optimized for detection quality (may reduce FPS)

See [`config/`](config/) directory for all available options.

## 🔧 Command Line Options

```bash
python3 src/main.py [OPTIONS]

Options:
  --width INTEGER       Camera frame width (default: 480)
  --height INTEGER      Camera frame height (default: 270) 
  --fps INTEGER         Camera FPS (default: 30)
  --model TEXT          Path to YOLO model (default: yolov8n.pt)
  --confidence FLOAT    Detection confidence threshold (default: 0.5)
  --headless            Run without GUI, save frames to disk
  --output-dir TEXT     Output directory for headless mode (default: /tmp/output)
  --help               Show this message and exit
```

### Usage Examples

```bash
# Standard operation (GUI mode)
python3 src/main.py

# Web-based live viewer
python3 src/web_viewer.py
# Then open browser to http://localhost:5000

# High-performance optimized mode
python3 src/high_performance_viewer.py

# High-confidence detections only
python3 src/main.py --confidence 0.8

# Headless operation with custom output
python3 src/main.py --headless --output-dir ./captures

# Performance mode (smaller resolution)
python3 src/main.py --width 424 --height 240

# Custom model
python3 src/main.py --model ./models/custom_yolo.pt
```

## 🌐 Web Viewer Mode

Access live detection through your browser:

```bash
# Start web viewer
python3 src/web_viewer.py --width 640 --height 480

# Open browser to: http://localhost:5000
# Features:
# - Live video stream with detection overlays
# - Real-time FPS and statistics
# - Confidence threshold adjustment
# - Responsive web interface
```

## ⚡ High-Performance Mode

For maximum FPS on Pi 5:

```bash
# Start high-performance pipeline
python3 src/high_performance_viewer.py

# Optimizations included:
# - 320x320 inference resolution (vs 640x480 display)
# - Adaptive frame skipping
# - Fast JPEG encoding
# - Confidence filtering (0.6+ threshold)
# - Optimized detection loop
```

## 🛠 Troubleshooting

### System Diagnostics
```bash
# Run comprehensive system check
./scripts/system_check.sh

# Performance benchmarking
./scripts/benchmark.sh
```

For detailed troubleshooting, see [`docs/troubleshooting.md`](docs/troubleshooting.md)

### Camera Issues

**Camera not detected:**
```bash
# Check USB connection
lsusb | grep Intel

# Expected output:
# Bus 003 Device 002: ID 8086:0b07 Intel Corp. RealSense D435

# Check udev rules
ls -la /etc/udev/rules.d/99-realsense-libusb.rules

# Reload rules if needed
sudo udevadm control --reload-rules && sudo udevadm trigger
```

**USB bandwidth issues:**
```bash
# Check USB speed
dmesg | grep -i "usb.*high\|usb.*super"

# Reduce camera resolution if bandwidth limited
python3 src/main.py --width 320 --height 240 --fps 15
```

**Power issues:**
```bash
# Check power supply (requires 5V/5A for Pi 5 + camera)
vcgencmd get_throttled

# 0x0 = OK, anything else indicates power issues
```

### Performance Issues

**Low FPS (< 2 FPS):**
1. Use high-performance mode: `python3 src/high_performance_viewer.py`
2. Reduce resolution: `--width 424 --height 240`
3. Lower frame rate: `--fps 15`
4. Increase confidence threshold: `--confidence 0.7`
5. Check CPU temperature: `vcgencmd measure_temp`
6. Try web viewer for better performance: `python3 src/web_viewer.py`

**High memory usage:**
```bash
# Check memory usage
free -h

# If swap is being used heavily, consider:
# 1. Reducing batch size in model
# 2. Using lower resolution
# 3. Adding more swap space
```

**Model loading errors:**
```bash
# Download YOLOv8-nano manually
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Check available disk space
df -h
```

### System Issues

**Permission denied for camera:**
```bash
# Add user to plugdev group
sudo usermod -a -G plugdev $USER

# Log out and back in, then verify:
groups | grep plugdev
```

**Import errors:**
```bash
# Reinstall dependencies
source venv/bin/activate
pip install --upgrade -r requirements.txt

# Check Python version
python3 --version  # Should be >= 3.11
```

## 🧪 Testing

Run the test suite to validate installation and performance:

```bash
# Activate environment
source venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run only FPS tests
python -m pytest tests/test_fps.py -v

# Run performance benchmarks
python -m pytest tests/test_fps.py::TestPipelinePerformance -v
```

## 🔄 Performance Optimization Strategies

### Implemented Optimizations

**High-Performance Pipeline (`src/high_performance_viewer.py`):**
- **Dual-resolution processing:** 320×320 inference, 640×480 display
- **Adaptive frame skipping:** Skip frames when processing is slow
- **Fast encoding:** Optimized JPEG compression for web streaming
- **Confidence filtering:** Only show detections above 0.6 threshold
- **Efficient memory management:** Reduced allocations and copies

**Web Viewer Optimizations (`src/web_viewer.py`):**
- **Flask streaming:** Real-time video stream to browser
- **Background processing:** Non-blocking inference pipeline
- **Responsive interface:** Adjustable confidence thresholds
- **Statistics display:** Live FPS and performance metrics

### Advanced Optimization Options

**ONNX Backend (Future):**
```bash
# Export YOLOv8 to ONNX format for faster inference
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', simplify=True)
"
# Then use: model = YOLO('yolov8n.onnx')
```

**NCNN Quantization (Future):**
```bash
# Convert to NCNN int8 for maximum Pi 5 performance
# Requires ncnn-python and quantization dataset
# See: https://github.com/ultralytics/ultralytics/blob/main/docs/modes/export.md
```

**System-Level Optimizations:**
```bash
# Set CPU governor to performance mode
sudo cpufreq-set -g performance

# Increase GPU memory split
sudo raspi-config  # Advanced Options -> Memory Split -> 128

# Optimize swap settings
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

## 📚 Documentation

- [**Installation Guide**](docs/installation_guide.md) - Detailed setup instructions
- [**Performance Optimization**](docs/performance_optimization.md) - Advanced tuning strategies  
- [**Troubleshooting**](docs/troubleshooting.md) - Common issues and solutions
- [**Development Log**](docs/development_log.md) - Complete project timeline
- [**API Reference**](docs/api_reference.md) - Code documentation (TODO)

### Quick Links

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Intel RealSense SDK](https://dev.intelrealsense.com/)
- [Raspberry Pi 5 Optimization Guide](https://www.raspberrypi.org/documentation/)
- [OpenCV Optimization](https://docs.opencv.org/4.x/db/d05/tutorial_config_reference.html)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes with tests
4. Update the development log in `steps.md`
5. Submit a pull request

## 📄 License

This project is open source. Please ensure compliance with:
- Ultralytics YOLOv8 License
- Intel RealSense SDK License
- OpenCV License

---

**System Requirements Summary:**
- Raspberry Pi 5 (8GB RAM)
- Ubuntu 24.04 LTS  
- Intel RealSense D435 camera
- Python ≥ 3.11
- 5V/5A power supply
- USB 3.0 connection for camera
