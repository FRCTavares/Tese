#!/usr/bin/env python3
"""
Launch file for camera node only.
Useful for testing camera functionality independently.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for camera node only."""
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_width',
            default_value='480',
            description='Camera frame width'
        ),
        
        DeclareLaunchArgument(
            'camera_height',
            default_value='270',
            description='Camera frame height'
        ),
        
        DeclareLaunchArgument(
            'camera_fps',
            default_value='30',
            description='Camera frames per second'
        ),
        
        DeclareLaunchArgument(
            'enable_depth',
            default_value='false',
            description='Enable depth camera stream'
        ),
        
        Node(
            package='uav_object_detection',
            executable='camera_node.py',
            name='camera_node',
            parameters=[{
                'width': LaunchConfiguration('camera_width'),
                'height': LaunchConfiguration('camera_height'),
                'fps': LaunchConfiguration('camera_fps'),
                'enable_depth': LaunchConfiguration('enable_depth'),
            }],
            output='screen'
        )
    ])
