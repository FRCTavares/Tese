# ROS2 UAV Object Detection Package

This ROS2 package provides a modular object detection system for UAV applications using YOLO models and Intel RealSense cameras.

## Architecture

The system consists of three main nodes:

### 1. Camera Node (`camera_node.py`)
- **Purpose**: Captures and publishes camera frames
- **Hardware**: Intel RealSense D435 (with fallback to mock camera)
- **Publishers**:
  - `/camera/color/image_raw` (sensor_msgs/Image)
  - `/camera/color/camera_info` (sensor_msgs/CameraInfo)
  - `/camera/depth/image_raw` (sensor_msgs/Image) - optional
  - `/camera/depth/camera_info` (sensor_msgs/CameraInfo) - optional

### 2. Detector Node (`detector_node.py`)
- **Purpose**: Performs object detection on camera images
- **AI Models**: YOLOv5, YOLOv8, YOLOv10 support
- **Subscribers**:
  - `/camera/color/image_raw` (sensor_msgs/Image)
- **Publishers**:
  - `/detections` (uav_object_detection/DetectionArray)
  - `/detections/annotated_image` (sensor_msgs/Image)

### 3. Visualization Node (`visualization_node.py`)
- **Purpose**: Displays detection results and performance metrics
- **Subscribers**:
  - `/detections/annotated_image` (sensor_msgs/Image)
  - `/detections` (uav_object_detection/DetectionArray)
- **Features**: Live display, video recording, performance overlay

## Custom Messages

### Detection.msg
```
Header header
uint32 class_id
string class_name
float32 confidence
geometry_msgs/Point center
geometry_msgs/Point bbox_min
geometry_msgs/Point bbox_max
float32 area
```

### DetectionArray.msg
```
Header header
Detection[] detections
uint32 total_detections
float32 inference_time_ms
string model_name
```

## Installation

### Prerequisites
```bash
# ROS2 (Humble or newer)
sudo apt update
sudo apt install ros-humble-desktop

# Python dependencies
pip install ultralytics opencv-python pyrealsense2

# ROS2 Python packages
sudo apt install ros-humble-cv-bridge ros-humble-vision-msgs
```

### Build Package
```bash
# Create workspace (if not exists)
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Clone or copy this package
cp -r /path/to/ROS_UAV_Detection ./uav_object_detection

# Build
cd ~/ros2_ws
colcon build --packages-select uav_object_detection

# Source workspace
source install/setup.bash
```

## Usage

### Quick Start
```bash
# Launch complete system
ros2 launch uav_object_detection uav_detection.launch.py

# Launch with custom model
ros2 launch uav_object_detection uav_detection.launch.py model_path:=yolov8s.pt

# Launch camera only
ros2 launch uav_object_detection camera_only.launch.py
```

### Launch with Configurations
```bash
# High performance (low latency)
ros2 launch uav_object_detection uav_detection.launch.py \
    camera_width:=320 camera_height:=240 camera_fps:=60 \
    model_path:=yolov8n.pt device:=cuda

# High quality (high accuracy)
ros2 launch uav_object_detection uav_detection.launch.py \
    camera_width:=640 camera_height:=480 camera_fps:=15 \
    model_path:=yolov8s.pt enable_depth:=true
```

### Using Configuration Files
```bash
# Load default parameters
ros2 launch uav_object_detection uav_detection.launch.py \
    --ros-args --params-file config/default_params.yaml

# Load high performance config
ros2 launch uav_object_detection uav_detection.launch.py \
    --ros-args --params-file config/high_performance.yaml
```

### Manual Node Launch
```bash
# Terminal 1: Camera
ros2 run uav_object_detection camera_node.py

# Terminal 2: Detector
ros2 run uav_object_detection detector_node.py \
    --ros-args -p model_path:=yolov8n.pt

# Terminal 3: Visualization
ros2 run uav_object_detection visualization_node.py
```

## Configuration Parameters

### Camera Node Parameters
- `width`: Frame width (default: 480)
- `height`: Frame height (default: 270)
- `fps`: Camera FPS (default: 30)
- `enable_depth`: Enable depth stream (default: false)
- `publish_rate`: Publishing rate (default: 30.0)

### Detector Node Parameters
- `model_path`: Path to YOLO model (default: "yolov8n.pt")
- `confidence_threshold`: Detection confidence (default: 0.5)
- `nms_threshold`: NMS threshold (default: 0.4)
- `input_size`: Model input size (default: 640)
- `device`: Inference device (default: "cpu")
- `max_detections`: Maximum detections per frame (default: 100)

### Visualization Node Parameters
- `display_window`: Show live window (default: true)
- `save_video`: Save detection video (default: false)
- `video_output_path`: Video save path (default: "/tmp/detections.mp4")
- `fps`: Video FPS (default: 30)

## Performance Optimization

### For High FPS (Low Latency)
1. Use smaller input resolution (320x240)
2. Use YOLOv8n or YOLOv5n models
3. Set smaller `input_size` (416)
4. Disable depth stream
5. Use GPU (`device: cuda`)

### For High Accuracy
1. Use larger input resolution (640x480 or higher)
2. Use YOLOv8s or YOLOv8m models
3. Set larger `input_size` (800)
4. Lower `confidence_threshold` (0.3)
5. Enable depth stream for context

## Monitoring and Debugging

### View Topics
```bash
# List all topics
ros2 topic list

# Monitor detection rate
ros2 topic hz /detections

# View detection messages
ros2 topic echo /detections

# Monitor camera stream
ros2 topic hz /camera/color/image_raw
```

### Performance Monitoring
```bash
# Node resource usage
ros2 node info /detector_node

# System performance
htop
nvtop  # for GPU monitoring
```

### Debugging
```bash
# Enable debug logging
ros2 run uav_object_detection detector_node.py \
    --ros-args --log-level DEBUG

# Record data for analysis
ros2 bag record /camera/color/image_raw /detections
```

## Integration with UAV Systems

### ROS2 Integration
The package integrates seamlessly with other ROS2 UAV packages:

```bash
# Example with PX4 autopilot
ros2 launch px4 px4.launch.py
ros2 launch uav_object_detection uav_detection.launch.py

# Example with ArduPilot
ros2 launch ardupilot_sitl sitl.launch.py
ros2 launch uav_object_detection uav_detection.launch.py
```

### Custom Integration
Create custom nodes that subscribe to `/detections` for:
- Object tracking
- Navigation decisions
- Mission planning
- Payload control

## Hardware Requirements

### Minimum Requirements
- CPU: ARM Cortex-A76 (Raspberry Pi 5) or x86_64
- RAM: 4GB
- Storage: 16GB
- Camera: USB camera or Intel RealSense D435

### Recommended for Real-time Performance
- CPU: Intel i5 or equivalent ARM with GPU
- RAM: 8GB+
- GPU: NVIDIA with CUDA support
- Storage: SSD 32GB+
- Camera: Intel RealSense D435 or D455

## Model Support

### Supported YOLO Variants
- **YOLOv5**: n, s, m, l, x variants
- **YOLOv8**: n, s, m, l, x variants  
- **YOLOv10**: n, s, m, l, x variants

### Model Files
Place model files in a models directory:
```
~/models/
├── yolov8n.pt
├── yolov8s.pt
├── yolov8n.onnx
└── yolov8s.onnx
```

## Contributing

1. Follow ROS2 coding standards
2. Add unit tests for new features
3. Update documentation
4. Test on target hardware

## License

See LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review ROS2 logs
3. Open an issue on the repository
