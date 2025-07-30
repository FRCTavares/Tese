# Performance Optimization Guide

Advanced strategies for optimizing the real-time object detection pipeline on Raspberry Pi 5.

## Current Performance Status

### Baseline Performance (Pi 5, 8GB RAM)
- **Standard Mode**: ~1.9 FPS (424×240)
- **Web Viewer**: ~2.5 FPS (640×480 streaming)
- **High-Performance**: ~3.5+ FPS (with optimizations)
- **Target Goal**: 5+ FPS

## Implemented Optimizations

### 1. High-Performance Pipeline (`src/high_performance_viewer.py`)

**Dual-Resolution Processing:**
- Display resolution: 640×480 (clear viewing)
- Inference resolution: 320×320 (fast processing)
- **Benefit**: ~40% faster inference

**Adaptive Frame Skipping:**
- Skip frames when processing is slow
- Dynamic adjustment based on performance
- **Benefit**: Maintains responsiveness under load

**Fast JPEG Encoding:**
- Quality: 85% (vs 95% default)
- **Benefit**: ~30% faster encoding

**Confidence Filtering:**
- Threshold: 0.6 (vs 0.5 default)
- **Benefit**: Fewer false positives to process

### 2. Web Viewer Optimizations (`src/web_viewer.py`)

**Flask Streaming:**
- MJPEG streaming to browser
- Background processing threads
- **Benefit**: Remote access, better resource management

**Memory Management:**
- Reduced frame copying
- Buffer reuse
- **Benefit**: Lower memory usage, stable performance

## Advanced Optimization Strategies

### Model-Level Optimizations

#### 1. ONNX Backend Export
```python
from ultralytics import YOLO

# Export YOLOv8 to ONNX format
model = YOLO('yolov8n.pt')
model.export(
    format='onnx',
    imgsz=480,
    optimize=True,
    simplify=True,
    dynamic=False,
    opset=12,
    half=False
)

# Use ONNX model for faster inference
detector = Detector(model_path="yolov8n.onnx", backend="onnx")
```

**Expected improvement**: 20-30% faster inference

#### 2. Model Pruning
```bash
# Prune 30% of model parameters
yolo train model=yolov8n.pt prune=0.3 data=custom_dataset.yaml epochs=100

# Or use Ultralytics pruning utilities
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.model = torch.nn.utils.prune.global_unstructured(
    model.model.parameters(), pruning_method=torch.nn.utils.prune.L1Unstructured, amount=0.3
)
model.save('yolov8n_pruned.pt')
"
```

**Expected improvement**: 30-50% faster inference, 40% smaller model

#### 3. NCNN int8 Quantization
```bash
# Install NCNN tools
git clone https://github.com/Tencent/ncnn.git
cd ncnn && mkdir build && cd build
cmake .. && make -j4

# Convert ONNX to NCNN
./tools/onnx/onnx2ncnn yolov8n.onnx yolov8n.param yolov8n.bin

# Quantize to int8 (requires calibration data)
./tools/quantize/ncnn2int8 yolov8n.param yolov8n.bin yolov8n_int8.param yolov8n_int8.bin calibration_images/
```

**Expected improvement**: 2-4x faster inference, 4x smaller model

### System-Level Optimizations

#### 1. CPU Governor Settings
```bash
# Set to performance mode
sudo cpufreq-set -g performance

# Verify setting
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Make persistent
echo 'GOVERNOR="performance"' | sudo tee -a /etc/default/cpufrequtils
```

#### 2. GPU Memory Split
```bash
# Increase GPU memory (helps with OpenCV operations)
sudo raspi-config
# Advanced Options -> Memory Split -> 128

# Or edit directly
echo 'gpu_mem=128' | sudo tee -a /boot/firmware/config.txt
sudo reboot
```

#### 3. USB Bandwidth Optimization
```bash
# Check USB tree
lsusb -t

# Ensure camera is on USB 3.0 port
# Pi 5 has 2x USB 3.0 and 2x USB 2.0 ports
# Use blue USB 3.0 ports for camera

# Check USB speed
dmesg | grep -i "usb.*super"
```

#### 4. Memory and Swap Tuning
```bash
# Optimize swap usage
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf

# Increase I/O scheduler performance
echo 'mq-deadline' | sudo tee /sys/block/mmcblk0/queue/scheduler
```

### Input/Camera Optimizations

#### 1. Resolution Strategy
```python
# Optimal resolutions for Pi 5 performance
PERFORMANCE_CONFIGS = {
    'max_fps': {'width': 424, 'height': 240, 'fps': 30},      # ~5+ FPS
    'balanced': {'width': 640, 'height': 480, 'fps': 30},     # ~3 FPS  
    'quality': {'width': 1280, 'height': 720, 'fps': 15},     # ~1 FPS
}
```

#### 2. Camera Buffer Management
```python
# Optimize RealSense settings
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

# Disable auto-exposure for consistent performance
color_sensor = device.first_color_sensor()
color_sensor.set_option(rs.option.enable_auto_exposure, False)
color_sensor.set_option(rs.option.exposure, 166)  # Fixed exposure

# Reduce frame queue size
pipeline.start(config)
```

### Multi-Threading Optimizations

#### 1. Camera and Inference Threading
```python
import threading
from queue import Queue

class OptimizedPipeline:
    def __init__(self):
        self.frame_queue = Queue(maxsize=2)  # Small buffer
        
    def camera_thread(self):
        """Dedicated thread for camera capture"""
        while self.running:
            frame = self.camera.get_frame()
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
                
    def inference_thread(self):
        """Dedicated thread for inference"""
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                detections = self.detector.detect(frame)
                self.display_queue.put((frame, detections))
```

#### 2. OpenCV Threading
```python
# Enable OpenCV threading
cv2.setUseOptimized(True)
cv2.setNumThreads(4)  # Use all Pi 5 cores

# Check OpenCV build info
print(cv2.getBuildInformation())
```

## Performance Monitoring

### 1. Real-Time Monitoring Script
```python
# Create monitoring script
import psutil
import time

def monitor_performance():
    while True:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        temperature = psutil.sensors_temperatures()
        
        print(f"CPU: {cpu_percent}% | Memory: {memory.percent}% | Temp: {temperature}")
        time.sleep(5)
```

### 2. Benchmark Script
```bash
# Run FPS benchmark
python3 tests/test_fps.py::TestPipelinePerformance::test_end_to_end_performance -v

# System stress test
stress-ng --cpu 4 --timeout 60s
# Monitor while running detection pipeline
```

## Model Selection & Benchmarking

### Multi-Model Performance Comparison

The project includes comprehensive model benchmarking tools to help you choose the optimal model for your specific use case:

```bash
# Download all available models (YOLOv5, YOLOv8, YOLOv10 in multiple formats)
python scripts/download_models.py

# Run comprehensive benchmark comparing all models
./scripts/multi_model_benchmark.sh

# Quick comparison of nano models only
./scripts/multi_model_benchmark.sh --quick
```

### Model Selection Guidelines

**For Real-Time Applications (≥10 FPS):**
- Primary: `yolov8n.onnx`, `yolov10n.onnx`
- Alternative: `yolov5n.pt`, `yolov8n.pt`

**For Interactive Applications (5-10 FPS):**
- Primary: `yolov8s.onnx`, `yolov10s.onnx`
- Alternative: `yolov5s.pt`, `yolov8s.pt`

**For High Accuracy (Batch Processing):**
- Primary: `yolov8m.onnx`
- Alternative: `yolov8m.pt`

### Format Optimization

**ONNX vs PyTorch Performance:**
- ONNX models typically provide 10-30% performance improvement
- Smaller memory footprint
- Better cross-platform compatibility
- Optimized inference engines

**Quantization Options:**
```bash
# Create quantized TensorFlow Lite model
python -c "from ultralytics import YOLO; YOLO('models/yolov8n.pt').export(format='tflite', int8=True)"

# Create quantized ONNX model (requires onnxruntime)
python -c "
import onnx
from onnxruntime.quantization import quantize_dynamic
quantize_dynamic('models/yolov8n.onnx', 'models/yolov8n_quantized.onnx')
"
```

### Benchmark Results Analysis

The benchmarking suite provides detailed analysis including:
- FPS performance across all models
- Memory usage patterns
- CPU utilization
- Inference time statistics
- Efficiency metrics (FPS per MB)
- Format comparison (PyTorch vs ONNX)
- Model family comparison (YOLOv5 vs YOLOv8 vs YOLOv10)

Results are saved in multiple formats:
- `benchmark_results/multi_model_benchmark_YYYYMMDD_HHMMSS.json` (detailed results)
- `benchmark_results/benchmark_comparison_YYYYMMDD_HHMMSS.csv` (analysis-ready)

## Performance Targets by Mode

| Mode | Target FPS | Resolution | Optimizations |
|------|------------|------------|---------------|
| Development | 1-2 FPS | 640×480 | Basic, reliable |
| Production | 3-4 FPS | 424×240 | High-performance pipeline |
| Optimized | 5+ FPS | 320×320 | ONNX + system tuning |
| Edge Cases | 8+ FPS | Custom | NCNN int8 + extreme tuning |

## Optimization Checklist

### Quick Wins (5-15 minutes)
- [ ] Use high-performance viewer: `python3 src/high_performance_viewer.py`
- [ ] Set CPU governor to performance: `sudo cpufreq-set -g performance`
- [ ] Use optimal camera resolution: 424×240 @ 30fps
- [ ] Increase confidence threshold: `--confidence 0.6`

### Medium Effort (1-2 hours)
- [ ] Export model to ONNX format
- [ ] Optimize swap and memory settings
- [ ] Configure GPU memory split
- [ ] Implement multi-threading

### Advanced (4+ hours)
- [ ] Model pruning and quantization
- [ ] NCNN int8 conversion
- [ ] Custom inference backend
- [ ] Hardware-specific optimizations

## Troubleshooting Performance Issues

### Low FPS (< 2 FPS)
1. Check CPU temperature: `vcgencmd measure_temp`
2. Verify CPU governor: `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`
3. Monitor memory usage: `free -h`
4. Check USB connection speed: `lsusb -t`

### High Memory Usage
1. Reduce camera buffer size
2. Lower inference resolution
3. Increase swap space
4. Monitor with: `watch -n 1 free -h`

### Thermal Throttling
1. Check temperature: `vcgencmd measure_temp`
2. Improve cooling (heatsink, fan)
3. Reduce CPU load
4. Monitor: `watch -n 1 vcgencmd measure_temp`

## Future Optimization Opportunities

1. **Custom YOLOv8 Architecture**: Reduce model complexity for specific use cases
2. **TensorRT Integration**: If switching to NVIDIA Jetson platform
3. **OpenVINO Support**: Intel-optimized inference framework
4. **ARM NN Integration**: Native ARM neural network library
5. **Edge TPU Support**: Google Coral integration for ultra-fast inference
