#!/bin/bash
# Setup script for ROS2 UAV Object Detection package

set -e

echo "=========================================="
echo "ROS2 UAV Object Detection Setup"
echo "=========================================="

# Check if ROS2 is installed
if ! command -v ros2 &> /dev/null; then
    echo "ERROR: ROS2 not found. Please install ROS2 first."
    echo "Visit: https://docs.ros.org/en/humble/Installation.html"
    exit 1
fi

echo "✓ ROS2 found"

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓ Python version: $python_version"

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Install ROS2 dependencies
echo "Installing ROS2 dependencies..."
sudo apt update
sudo apt install -y \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    ros-humble-image-transport \
    ros-humble-message-filters \
    ros-humble-launch \
    ros-humble-launch-ros

# Create workspace if it doesn't exist
if [ ! -d "~/ros2_ws" ]; then
    echo "Creating ROS2 workspace..."
    mkdir -p ~/ros2_ws/src
fi

# Copy package to workspace
echo "Setting up package in workspace..."
PACKAGE_DIR="$HOME/ros2_ws/src/uav_object_detection"

if [ ! -d "$PACKAGE_DIR" ]; then
    mkdir -p "$PACKAGE_DIR"
fi

# Copy files (assuming script is run from package directory)
cp -r . "$PACKAGE_DIR/"

# Build package
echo "Building package..."
cd ~/ros2_ws
colcon build --packages-select uav_object_detection

# Source workspace
echo "Sourcing workspace..."
source install/setup.bash

# Make scripts executable
chmod +x "$PACKAGE_DIR/src/"*.py
chmod +x "$PACKAGE_DIR/launch/"*.py

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To use the package:"
echo "1. Source your workspace:"
echo "   source ~/ros2_ws/install/setup.bash"
echo ""
echo "2. Launch the detection system:"
echo "   ros2 launch uav_object_detection uav_detection.launch.py"
echo ""
echo "3. Or launch individual components:"
echo "   ros2 run uav_object_detection camera_node.py"
echo "   ros2 run uav_object_detection detector_node.py"
echo "   ros2 run uav_object_detection visualization_node.py"
echo ""
echo "For more information, see README.md"
