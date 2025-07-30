#!/bin/bash
# Intel RealSense SDK Installation for Raspberry Pi 5
# This script builds and installs the RealSense SDK from source

set -e

echo "=== Intel RealSense SDK Installation for Raspberry Pi 5 ==="

# Update packages
sudo apt update

# Install dependencies for building RealSense SDK
echo "Installing build dependencies..."
sudo apt install -y \
    git \
    libssl-dev \
    libusb-1.0-0-dev \
    libudev-dev \
    pkg-config \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    cmake \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools

# Clone Intel RealSense SDK
echo "Cloning Intel RealSense SDK..."
cd ~
if [ -d "librealsense" ]; then
    echo "librealsense directory exists, updating..."
    cd librealsense
    git pull
else
    git clone https://github.com/IntelRealSense/librealsense.git
    cd librealsense
fi

# Check out stable version
git checkout v2.54.1

# Install udev rules
echo "Installing udev rules..."
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

# Create build directory
mkdir -p build && cd build

# Configure build with Python bindings
echo "Configuring build..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=true \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DBUILD_EXAMPLES=false \
    -DBUILD_GRAPHICAL_EXAMPLES=false

# Build (this will take 30-60 minutes on Pi 5)
echo "Building Intel RealSense SDK... This may take 30-60 minutes on Pi 5"
make -j4

# Install
echo "Installing Intel RealSense SDK..."
sudo make install

# Update library path
echo "/usr/local/lib" | sudo tee -a /etc/ld.so.conf.d/99-realsense.conf
sudo ldconfig

echo "Intel RealSense SDK installation complete!"
echo "Testing installation..."

# Test installation
python3 -c "import pyrealsense2 as rs; print(f'pyrealsense2 version: {rs.__version__}')" || {
    echo "Installation verification failed"
    exit 1
}

echo "✓ Intel RealSense SDK successfully installed and verified"
