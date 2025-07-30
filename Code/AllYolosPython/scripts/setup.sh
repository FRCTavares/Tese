#!/bin/bash
# Setup script for Real-Time Object Detection Pipeline on Raspberry Pi 5
# Handles system prerequisites, udev rules, and Python environment setup

set -e  # Exit on any error

echo "=== Raspberry Pi 5 Object Detection Pipeline Setup ==="
echo "Setting up system prerequisites for Intel RealSense D435 camera..."

# Check if running on supported system
if ! grep -q "Ubuntu" /etc/os-release; then
    echo "Warning: This script is optimized for Ubuntu 24.04. Proceed with caution on other distributions."
fi

# Update system packages
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies for RealSense and OpenCV
echo "Installing system dependencies..."
sudo apt install -y \
    build-essential \
    cmake \
    git \
    libssl-dev \
    libusb-1.0-0-dev \
    pkg-config \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    usbutils \
    v4l-utils

# Setup udev rules for Intel RealSense D435
echo "Setting up udev rules for Intel RealSense D435..."
sudo tee /etc/udev/rules.d/99-realsense-libusb.rules > /dev/null << 'EOF'
# Intel RealSense D435 udev rules
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b07", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b3a", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad1", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad2", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad3", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad4", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b5c", MODE="0666", GROUP="plugdev"
EOF

# Add user to plugdev group
sudo usermod -a -G plugdev $USER

# Reload udev rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# Check and optimize swap for Pi 5 (8GB RAM should be sufficient, but small swap helps)
CURRENT_SWAP=$(swapon --show=SIZE --noheadings --bytes | head -1)
if [ -z "$CURRENT_SWAP" ] || [ "$CURRENT_SWAP" -lt 1073741824 ]; then  # Less than 1GB
    echo "Setting up swap file for memory optimization..."
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
fi

# Set up Python virtual environment
echo "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
echo "Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# Install requirements with Pi 5 optimizations
pip install -r requirements.txt

# Verify RealSense installation
echo "Verifying Intel RealSense installation..."
python3 -c "import pyrealsense2 as rs; print(f'pyrealsense2 version: {rs.__version__}')" || {
    echo "Error: pyrealsense2 installation failed"
    exit 1
}

# Verify OpenCV installation
echo "Verifying OpenCV installation..."
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')" || {
    echo "Error: OpenCV installation failed"
    exit 1
}

# Verify YOLO installation
echo "Verifying Ultralytics YOLO installation..."
python3 -c "import ultralytics; print('Ultralytics YOLO installed successfully')" || {
    echo "Error: Ultralytics installation failed"
    exit 1
}

# Create output directory for headless mode
mkdir -p /tmp/output

echo ""
echo "=== Setup Complete ==="
echo "Please log out and log back in for group changes to take effect."
echo "To activate the Python environment, run: source venv/bin/activate"
echo "To test the camera, run: python3 src/main.py"
echo "For headless operation, run: python3 src/main.py --headless"
echo ""
echo "Troubleshooting:"
echo "- If camera not detected, check USB 3.0 connection and power"
echo "- Run 'lsusb | grep Intel' to verify camera detection"
echo "- Check /var/log/syslog for USB bandwidth issues"
