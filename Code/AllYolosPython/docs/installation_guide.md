# Installation Guide

Complete installation guide for the Real-Time Object Detection Pipeline on Raspberry Pi 5.

## Prerequisites

### Hardware Requirements
- **Raspberry Pi 5** (8GB RAM recommended)
- **Intel RealSense D435** depth-RGB camera
- **USB 3.0 port** for camera connection
- **5V/5A power supply** (official Pi 5 adapter recommended)
- **MicroSD card** (32GB+ Class 10 or better)

### Software Requirements
- **Ubuntu 24.04 LTS** (recommended OS)
- **Python ≥ 3.11**
- **Git** for source code management
- **USB 3.0 drivers** (usually included in Ubuntu)

## Installation Steps

### 1. System Setup

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Clone the repository
git clone <repository-url>
cd real-time-object-detection-pi5

# Make scripts executable
chmod +x scripts/*.sh
```

### 2. Basic Dependencies

```bash
# Run basic system setup
./scripts/setup.sh
```

This script will:
- Install system dependencies
- Set up udev rules for RealSense camera
- Configure Python virtual environment
- Install base Python packages
- Optimize system settings for Pi 5

### 3. Intel RealSense SDK

```bash
# Build and install RealSense SDK (30-60 minutes)
./scripts/setup_realsense.sh
```

This script will:
- Install build dependencies
- Clone and build Intel RealSense SDK v2.54.1
- Install system-wide with Python bindings
- Configure library paths

### 4. Virtual Environment Fix

```bash
# Fix RealSense in virtual environment
./scripts/fix_realsense_venv.sh
```

This script will:
- Link system pyrealsense2 to virtual environment
- Set up proper library paths
- Verify installation

### 5. Environment Activation

```bash
# Set library path and activate environment
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
source venv/bin/activate
```

Add to `~/.bashrc` for persistence:
```bash
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

## Verification

### Test Camera Connection
```bash
# Check camera detection
lsusb | grep Intel
# Expected: Bus XXX Device XXX: ID 8086:0b07 Intel Corp. RealSense D435

# Test camera streaming
python3 src/main.py --headless --width 424 --height 240
```

### Test Object Detection
```bash
# GUI mode (requires display)
python3 src/main.py --width 640 --height 480

# Web viewer mode
python3 src/web_viewer.py
# Open browser to: http://localhost:5000

# High-performance mode
python3 src/high_performance_viewer.py
```

## Alternative Installation Options

### Without RealSense Camera
If you don't have a RealSense camera, you can test with mock camera:

```bash
# Install without RealSense dependencies
pip install -r requirements/requirements_pi5.txt

# Test with mock camera
python3 src/main.py --width 640 --height 480
```

### Development Installation
For development with additional tools:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run code formatting
black src/ tests/

# Run type checking
mypy src/

# Run tests
pytest tests/ -v
```

## Post-Installation Configuration

### Performance Optimization
```bash
# Set CPU governor to performance
sudo cpufreq-set -g performance

# Increase GPU memory split
sudo raspi-config
# Advanced Options -> Memory Split -> 128
```

### Camera Permissions
```bash
# Add user to plugdev group (if not done by setup script)
sudo usermod -a -G plugdev $USER

# Log out and back in for changes to take effect
```

### Swap Configuration
```bash
# Check current swap
swapon --show

# Configure swap for better performance
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

## Troubleshooting

See [troubleshooting.md](troubleshooting.md) for common issues and solutions.

## Next Steps

After successful installation:
1. Read the [performance optimization guide](performance_optimization.md)
2. Explore the [Jupyter notebooks](../notebooks/) for model training
3. Check the [API reference](api_reference.md) for custom development
4. Run performance tests: `pytest tests/test_fps.py -v`
