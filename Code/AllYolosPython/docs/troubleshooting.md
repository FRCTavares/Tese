# Troubleshooting Guide

Common issues and solutions for the Real-Time Object Detection Pipeline.

## Camera Issues

### Camera Not Detected

**Symptoms:**
- `lsusb | grep Intel` returns nothing
- Error: "No RealSense devices found"

**Solutions:**
```bash
# 1. Check physical connection
# Ensure camera is connected to USB 3.0 port (blue connector)

# 2. Check power supply
# Pi 5 requires 5V/5A for camera operation
vcgencmd get_throttled
# 0x0 = OK, anything else indicates power issues

# 3. Verify udev rules
ls -la /etc/udev/rules.d/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# 4. Check USB tree
lsusb -t
# Camera should appear on USB 3.0 bus

# 5. Reinstall udev rules
sudo ./scripts/setup_realsense.sh
```

### "Couldn't Resolve Requests" Error

**Symptoms:**
- Camera detected but streaming fails
- Error during pipeline start

**Solutions:**
```bash
# 1. Use supported resolutions only
python3 src/main.py --width 424 --height 240  # ✓ Supported
python3 src/main.py --width 640 --height 480  # ✓ Supported
python3 src/main.py --width 320 --height 240  # ✗ Not supported

# 2. Check USB bandwidth
# Ensure no other USB 3.0 devices are competing for bandwidth

# 3. Lower frame rate
python3 src/main.py --width 640 --height 480 --fps 15

# 4. Check camera permissions
groups | grep plugdev
# If missing: sudo usermod -a -G plugdev $USER && logout
```

### Poor Image Quality

**Symptoms:**
- Dark or overexposed images
- Inconsistent exposure

**Solutions:**
```python
# Manual exposure settings in camera.py
color_sensor.set_option(rs.option.enable_auto_exposure, False)
color_sensor.set_option(rs.option.exposure, 166)  # Adjust value
color_sensor.set_option(rs.option.gain, 64)       # Adjust gain
```

## Performance Issues

### Low FPS (< 2 FPS)

**Diagnosis:**
```bash
# Check CPU usage
htop

# Check temperature
vcgencmd measure_temp

# Check memory
free -h

# Check swap usage
swapon --show
```

**Solutions:**
```bash
# 1. Use high-performance mode
python3 src/high_performance_viewer.py

# 2. Reduce resolution
python3 src/main.py --width 424 --height 240

# 3. Increase confidence threshold
python3 src/main.py --confidence 0.7

# 4. Set performance governor
sudo cpufreq-set -g performance

# 5. Check for thermal throttling
vcgencmd measure_temp
# If > 80°C, improve cooling
```

### High Memory Usage

**Symptoms:**
- System becomes slow
- Swap usage increases
- Out of memory errors

**Solutions:**
```bash
# 1. Monitor memory usage
watch -n 1 free -h

# 2. Reduce batch processing
# Edit detector.py to process single frames

# 3. Increase swap space
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# 4. Use mock camera for testing
python3 src/main.py  # Will automatically use mock if RealSense unavailable
```

### Thermal Throttling

**Symptoms:**
- Performance degrades over time
- CPU temperature > 80°C
- `vcgencmd get_throttled` returns non-zero

**Solutions:**
```bash
# 1. Monitor temperature
watch -n 5 vcgencmd measure_temp

# 2. Improve cooling
# - Add heatsink
# - Add cooling fan
# - Improve case ventilation

# 3. Reduce CPU load
python3 src/main.py --confidence 0.8  # Higher threshold
python3 src/main.py --fps 15           # Lower frame rate

# 4. Check power supply
vcgencmd get_throttled
# Bit meanings:
# 0: Under-voltage detected
# 1: Arm frequency capped
# 2: Currently throttled
# 3: Soft temperature limit active
```

## Installation Issues

### pyrealsense2 Import Error

**Symptoms:**
- `ModuleNotFoundError: No module named 'pyrealsense2'`
- Works system-wide but not in virtual environment

**Solutions:**
```bash
# 1. Run the venv fix script
./scripts/fix_realsense_venv.sh

# 2. Manual fix
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
source venv/bin/activate

# 3. Create symbolic links manually
cd venv/lib/python3.11/site-packages/
ln -sf /usr/local/lib/python3.11/site-packages/pyrealsense2 .

# 4. Verify installation
python3 -c "import pyrealsense2 as rs; print(rs.__version__)"
```

### Permission Denied Errors

**Symptoms:**
- Camera access denied
- USB permission errors

**Solutions:**
```bash
# 1. Add user to plugdev group
sudo usermod -a -G plugdev $USER

# 2. Check group membership
groups

# 3. Logout and login again for changes to take effect

# 4. Verify udev rules
ls -la /etc/udev/rules.d/99-realsense-libusb.rules

# 5. Reload udev rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Package Installation Failures

**Symptoms:**
- pip install errors
- Missing system dependencies

**Solutions:**
```bash
# 1. Update package lists
sudo apt update

# 2. Install missing system dependencies
sudo apt install -y python3-dev build-essential cmake

# 3. Upgrade pip
pip install --upgrade pip setuptools wheel

# 4. Use alternative requirements file
pip install -r requirements/requirements_pi5.txt  # Without RealSense

# 5. Install packages individually
pip install opencv-python numpy ultralytics
```

## Model and Inference Issues

### Model Download Failures

**Symptoms:**
- "Failed to download model" errors
- Slow model loading

**Solutions:**
```bash
# 1. Pre-download model
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 2. Check internet connection
ping github.com

# 3. Use local model file
python3 src/main.py --model ./models/yolov8n.pt

# 4. Check disk space
df -h
```

### Poor Detection Accuracy

**Symptoms:**
- No objects detected
- Many false positives

**Solutions:**
```bash
# 1. Adjust confidence threshold
python3 src/main.py --confidence 0.3  # Lower for more detections
python3 src/main.py --confidence 0.8  # Higher for fewer false positives

# 2. Check lighting conditions
# Ensure adequate lighting for camera

# 3. Test with mock camera
# Verify pipeline works with synthetic data

# 4. Use web viewer for debugging
python3 src/web_viewer.py
# Open http://localhost:5000 to see live detections
```

### Slow Inference

**Symptoms:**
- Very low FPS
- Long processing times

**Solutions:**
```bash
# 1. Use ONNX model
# Export model: python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
python3 src/main.py --model yolov8n.onnx

# 2. Reduce input resolution
python3 src/main.py --width 424 --height 240

# 3. Use high-performance mode
python3 src/high_performance_viewer.py

# 4. Check CPU governor
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sudo cpufreq-set -g performance
```

## Web Viewer Issues

### Cannot Access Web Interface

**Symptoms:**
- Browser shows "This site can't be reached"
- Connection refused errors

**Solutions:**
```bash
# 1. Check if server is running
ps aux | grep python3

# 2. Verify port is available
netstat -tlnp | grep :5000

# 3. Check firewall settings
sudo ufw status

# 4. Try different port
python3 src/web_viewer.py --port 8080

# 5. Access from same machine first
curl http://localhost:5000
```

### Poor Web Streaming Performance

**Symptoms:**
- Choppy video in browser
- High latency

**Solutions:**
```bash
# 1. Reduce streaming quality
# Edit web_viewer.py, lower JPEG quality

# 2. Use wired connection instead of WiFi

# 3. Close other browser tabs/applications

# 4. Use high-performance mode
python3 src/high_performance_viewer.py
```

## Network and Remote Access Issues

### SSH Connection Problems

**Solutions:**
```bash
# 1. Enable SSH on Pi 5
sudo systemctl enable ssh
sudo systemctl start ssh

# 2. Check SSH status
sudo systemctl status ssh

# 3. Find Pi IP address
hostname -I

# 4. Connect from remote machine
ssh pi@<PI_IP_ADDRESS>
```

### File Transfer Issues

**Solutions:**
```bash
# 1. Use SCP for file transfer
scp file.txt pi@<PI_IP>:/home/pi/

# 2. Use rsync for directories
rsync -av src/ pi@<PI_IP>:/home/pi/real-time-object-detection/src/

# 3. Enable SMB sharing (optional)
sudo apt install samba
sudo smbpasswd -a pi
```

## System Diagnostic Commands

### Quick Health Check
```bash
#!/bin/bash
# System health check script

echo "=== System Health Check ==="

echo "Temperature:"
vcgencmd measure_temp

echo -e "\nThrottling status:"
vcgencmd get_throttled

echo -e "\nMemory usage:"
free -h

echo -e "\nDisk usage:"
df -h

echo -e "\nCPU governor:"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

echo -e "\nUSB devices:"
lsusb | grep -E "(Intel|RealSense)"

echo -e "\nPython packages:"
python3 -c "import cv2, numpy, ultralytics; print('OpenCV:', cv2.__version__, 'NumPy:', numpy.__version__, 'Ultralytics: OK')"

echo -e "\nRealSense test:"
python3 -c "import pyrealsense2 as rs; print('pyrealsense2:', rs.__version__)" 2>/dev/null || echo "RealSense not available"
```

### Performance Monitoring
```bash
# Real-time system monitoring
watch -n 1 'echo "CPU: $(vcgencmd measure_temp) | $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)Hz"; echo "Memory:"; free -h | grep Mem; echo "Load:"; uptime'
```

## Getting Help

### Collect Debug Information
```bash
# Create debug report
echo "=== Debug Report ===" > debug_report.txt
echo "Date: $(date)" >> debug_report.txt
echo "System: $(uname -a)" >> debug_report.txt
echo "Python: $(python3 --version)" >> debug_report.txt
echo "Temperature: $(vcgencmd measure_temp)" >> debug_report.txt
echo "Throttling: $(vcgencmd get_throttled)" >> debug_report.txt
echo "Memory: $(free -h)" >> debug_report.txt
echo "USB: $(lsusb)" >> debug_report.txt
echo "Packages:" >> debug_report.txt
pip list >> debug_report.txt
```

### Enable Verbose Logging
```python
# Add to main.py for debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
python3 src/main.py --width 424 --height 240 2>&1 | tee debug.log
```

### Community Resources
- [Intel RealSense Documentation](https://dev.intelrealsense.com/)
- [Ultralytics YOLOv8 Issues](https://github.com/ultralytics/ultralytics/issues)
- [Raspberry Pi Forums](https://www.raspberrypi.org/forums/)
- [OpenCV Documentation](https://docs.opencv.org/)

### Reporting Issues
When reporting issues, include:
1. Hardware setup (Pi 5 model, camera model, power supply)
2. Software versions (OS, Python, package versions)
3. Complete error messages
4. Steps to reproduce
5. Debug report (see above)
6. Expected vs actual behavior
