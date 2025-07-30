#!/bin/bash
# System health check and diagnostic script for the detection pipeline

set -e

echo "=== Real-Time Object Detection Pipeline - System Check ==="
echo "Date: $(date)"
echo "System: $(uname -a)"
echo ""

# Function to check command success
check_command() {
    local cmd="$1"
    local desc="$2"
    
    echo -n "Checking $desc... "
    if eval "$cmd" >/dev/null 2>&1; then
        echo "✓ OK"
        return 0
    else
        echo "✗ FAILED"
        return 1
    fi
}

# System Health
echo "=== System Health ==="
echo "Temperature: $(vcgencmd measure_temp)"
echo "Throttling status: $(vcgencmd get_throttled)"
echo "CPU governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
echo "Memory usage:"
free -h
echo ""

# Disk space
echo "=== Disk Space ==="
df -h | grep -E "(Filesystem|/dev/root|/dev/mmcblk)"
echo ""

# Dependencies
echo "=== Dependencies Check ==="
check_command "python3 --version | grep -q '3.1[1-9]'" "Python 3.11+"
check_command "python3 -c 'import cv2'" "OpenCV"
check_command "python3 -c 'import numpy'" "NumPy"
check_command "python3 -c 'import ultralytics'" "Ultralytics YOLO"
check_command "python3 -c 'import flask'" "Flask"

# RealSense check
echo -n "Checking Intel RealSense... "
if python3 -c "import pyrealsense2 as rs; print('Version:', rs.__version__)" 2>/dev/null; then
    echo "✓ OK"
    REALSENSE_AVAILABLE=true
else
    echo "✗ Not available (will use mock camera)"
    REALSENSE_AVAILABLE=false
fi

# Camera hardware check
echo ""
echo "=== Hardware Check ==="
echo "USB devices:"
lsusb | grep -E "(Intel|RealSense)" || echo "No Intel RealSense devices found"

if [ "$REALSENSE_AVAILABLE" = true ]; then
    echo ""
    echo "RealSense device info:"
    python3 -c "
import pyrealsense2 as rs
ctx = rs.context()
devices = ctx.query_devices()
for dev in devices:
    print(f'Device: {dev.get_info(rs.camera_info.name)}')
    print(f'Serial: {dev.get_info(rs.camera_info.serial_number)}')
    print(f'Firmware: {dev.get_info(rs.camera_info.firmware_version)}')
" 2>/dev/null || echo "Could not query RealSense device info"
fi

# Performance test
echo ""
echo "=== Performance Test ==="
echo "Running quick inference test..."

cd "$(dirname "$0")/.."
source venv/bin/activate 2>/dev/null || echo "Virtual environment not found"

# Set library path
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Run quick test
python3 -c "
import sys
sys.path.append('src')
from detector import Detector
import numpy as np
import time

print('Initializing detector...')
detector = Detector()
detector.load_model()

print('Running inference test...')
test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

start_time = time.time()
detections = detector.detect(test_image)
inference_time = time.time() - start_time

print(f'Inference time: {inference_time:.3f}s')
print(f'Estimated FPS: {1.0/inference_time:.1f}')
print(f'Detections found: {len(detections)}')
" 2>/dev/null || echo "Performance test failed"

# Network check (for web viewer)
echo ""
echo "=== Network Check ==="
echo "Network interfaces:"
ip addr show | grep -E "inet.*scope global" | awk '{print $2}' || echo "No network interfaces found"

echo "Port 5000 availability:"
if netstat -tlnp 2>/dev/null | grep -q ":5000"; then
    echo "Port 5000 is in use"
else
    echo "Port 5000 is available"
fi

# Project structure check
echo ""
echo "=== Project Structure ==="
check_command "test -f src/main.py" "Main entry point"
check_command "test -f src/detector.py" "Detector module"
check_command "test -f src/camera.py" "Camera module"
check_command "test -f src/web_viewer.py" "Web viewer"
check_command "test -f src/high_performance_viewer.py" "High-performance viewer"
check_command "test -d models" "Models directory"
check_command "test -d tests" "Tests directory"

# Permissions check
echo ""
echo "=== Permissions Check ==="
echo "Current user groups:"
groups | tr ' ' '\n' | grep -E "(plugdev|video|dialout)" || echo "No relevant groups found"

echo "Udev rules:"
if [ -f /etc/udev/rules.d/99-realsense-libusb.rules ]; then
    echo "✓ RealSense udev rules installed"
else
    echo "✗ RealSense udev rules missing"
fi

# Final summary
echo ""
echo "=== Summary ==="
if [ "$REALSENSE_AVAILABLE" = true ]; then
    echo "✓ System ready for real camera detection"
    echo "Run: python3 src/main.py --width 424 --height 240"
else
    echo "⚠ System ready for mock camera detection only"
    echo "Run: python3 src/main.py --width 424 --height 240"
    echo "To install RealSense: ./scripts/setup_realsense.sh"
fi

echo ""
echo "For web viewer: python3 src/web_viewer.py"
echo "For high performance: python3 src/high_performance_viewer.py"
echo "For help: see docs/troubleshooting.md"

echo ""
echo "System check complete!"
