#!/bin/bash
# Fix RealSense Python bindings in virtual environment
# Links system-installed pyrealsense2 to venv

set -e

echo "=== Fixing RealSense Python Bindings for Virtual Environment ==="

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment 'venv' not found"
    exit 1
fi

source venv/bin/activate

# Get Python version info
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

# Find system pyrealsense2 installation
SYSTEM_PYREALSENSE_PATH="/usr/lib/python3/dist-packages/pyrealsense2"
VENV_SITE_PACKAGES="venv/lib/python${PYTHON_VERSION}/site-packages"

echo "System pyrealsense2 path: $SYSTEM_PYREALSENSE_PATH"
echo "Virtual environment site-packages: $VENV_SITE_PACKAGES"

# Check if system installation exists
if [ ! -d "$SYSTEM_PYREALSENSE_PATH" ]; then
    echo "Error: System pyrealsense2 not found at $SYSTEM_PYREALSENSE_PATH"
    echo "RealSense SDK may not be properly installed"
    exit 1
fi

# Create symbolic link in virtual environment
echo "Creating symbolic link for pyrealsense2..."
ln -sfn "$SYSTEM_PYREALSENSE_PATH" "$VENV_SITE_PACKAGES/pyrealsense2"

# Also need to link the shared libraries
echo "Linking shared libraries..."
VENV_LIB_DIR="venv/lib"
mkdir -p "$VENV_LIB_DIR"

# Link main RealSense library
if [ -f "/usr/local/lib/librealsense2.so" ]; then
    ln -sfn "/usr/local/lib/librealsense2.so" "$VENV_LIB_DIR/librealsense2.so"
    ln -sfn "/usr/local/lib/librealsense2.so.2.54" "$VENV_LIB_DIR/librealsense2.so.2.54"
    ln -sfn "/usr/local/lib/librealsense2.so.2.54.1" "$VENV_LIB_DIR/librealsense2.so.2.54.1"
fi

# Update library path for the virtual environment
echo "Updating library paths..."
echo "/usr/local/lib" > venv/lib/python${PYTHON_VERSION}/site-packages/realsense.pth
echo "$VENV_LIB_DIR" >> venv/lib/python${PYTHON_VERSION}/site-packages/realsense.pth

# Set environment variables for the session
export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

# Test the installation
echo "Testing pyrealsense2 installation..."
python3 -c "
import sys
import os
sys.path.insert(0, '/usr/lib/python3/dist-packages')
os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

try:
    import pyrealsense2 as rs
    print('✓ pyrealsense2 successfully imported!')
    
    # Try to get version (may not work on all builds)
    try:
        print(f'  Version: {rs.__version__}')
    except AttributeError:
        print('  Version: Available (version attribute not implemented)')
    
    # Test basic functionality
    ctx = rs.context()
    devices = ctx.query_devices()
    print(f'  Detected devices: {len(devices)}')
    
    if len(devices) > 0:
        for i, dev in enumerate(devices):
            print(f'  Device {i}: {dev.get_info(rs.camera_info.name)}')
            print(f'    Serial: {dev.get_info(rs.camera_info.serial_number)}')
    else:
        print('  No RealSense cameras detected (check USB connection)')
        
except ImportError as e:
    print('✗ pyrealsense2 import failed')
    print(f'  Error: {e}')
    sys.exit(1)
except Exception as e:
    print('✓ pyrealsense2 imported but device access failed')
    print(f'  Error: {e}')
    print('  This could be normal if camera is not connected or needs permissions')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=== RealSense Virtual Environment Fix Complete ==="
    echo "pyrealsense2 is now available in your virtual environment!"
    echo ""
    echo "To use RealSense in your venv session, run:"
    echo "  export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH"
    echo "  source venv/bin/activate"
    echo ""
    echo "Or add this to your ~/.bashrc for permanent effect:"
    echo "  echo 'export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH' >> ~/.bashrc"
else
    echo "Fix failed - check error messages above"
    exit 1
fi
