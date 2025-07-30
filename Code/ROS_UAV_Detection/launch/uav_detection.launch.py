#!/usr/bin/env python3
"""
Main launch file for UAV Object Detection system.
Launches camera, detector, and visualization nodes.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for UAV detection system."""
    
    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='yolov8n.pt',
        description='Path to YOLO model file'
    )
    
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.5',
        description='Confidence threshold for detections'
    )
    
    camera_width_arg = DeclareLaunchArgument(
        'camera_width',
        default_value='480',
        description='Camera frame width'
    )
    
    camera_height_arg = DeclareLaunchArgument(
        'camera_height',
        default_value='270',
        description='Camera frame height'
    )
    
    camera_fps_arg = DeclareLaunchArgument(
        'camera_fps',
        default_value='30',
        description='Camera frames per second'
    )
    
    enable_depth_arg = DeclareLaunchArgument(
        'enable_depth',
        default_value='false',
        description='Enable depth camera stream'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cpu',
        description='Device for inference (cpu, cuda, mps)'
    )
    
    display_window_arg = DeclareLaunchArgument(
        'display_window',
        default_value='true',
        description='Enable display window'
    )
    
    save_video_arg = DeclareLaunchArgument(
        'save_video',
        default_value='false',
        description='Save detection video'
    )
    
    # Camera Node
    camera_node = Node(
        package='uav_object_detection',
        executable='camera_node.py',
        name='camera_node',
        parameters=[{
            'width': LaunchConfiguration('camera_width'),
            'height': LaunchConfiguration('camera_height'),
            'fps': LaunchConfiguration('camera_fps'),
            'enable_depth': LaunchConfiguration('enable_depth'),
            'publish_rate': LaunchConfiguration('camera_fps'),
        }],
        output='screen'
    )
    
    # Detector Node
    detector_node = Node(
        package='uav_object_detection',
        executable='detector_node.py',
        name='detector_node',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'device': LaunchConfiguration('device'),
        }],
        output='screen'
    )
    
    # Visualization Node
    visualization_node = Node(
        package='uav_object_detection',
        executable='visualization_node.py',
        name='visualization_node',
        parameters=[{
            'display_window': LaunchConfiguration('display_window'),
            'save_video': LaunchConfiguration('save_video'),
        }],
        output='screen'
    )
    
    return LaunchDescription([
        # Launch arguments
        model_path_arg,
        confidence_threshold_arg,
        camera_width_arg,
        camera_height_arg,
        camera_fps_arg,
        enable_depth_arg,
        device_arg,
        display_window_arg,
        save_video_arg,
        
        # Log info
        LogInfo(msg="Starting UAV Object Detection System..."),
        
        # Nodes
        camera_node,
        detector_node,
        visualization_node,
        
        LogInfo(msg="UAV Object Detection System launched successfully!")
    ])
