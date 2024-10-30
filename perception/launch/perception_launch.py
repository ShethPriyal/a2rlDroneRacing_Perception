import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Get the share directory for the perception package
    perception_share = FindPackageShare('perception').find('perception')
    config_path = '/home/dhisum/Drone/a2rlDroneRacing/a2rlDroneRacing_main/install/perception/lib/perception/Detector/models/training_config.yaml'
    gp_model_path = '/home/dhisum/Drone/a2rlDroneRacing/a2rlDroneRacing_main/install/perception/lib/perception/Detector/models/gate_pose_estimator.json'
    d_weights_path = '/home/dhisum/Drone/a2rlDroneRacing/a2rlDroneRacing_main/install/perception/lib/perception/Detector/models/detector_weights.h5'
    gp_weights_path = '/home/dhisum/Drone/a2rlDroneRacing/a2rlDroneRacing_main/install/perception/lib/perception/Detector/models/gpesti_weights.h5'

    return LaunchDescription([
        DeclareLaunchArgument(
            'namespace',
            default_value='intel_aero',
            description='Namespace for the node'
        ),
        Node(
            package='perception',
            executable='detector_node.py',
            name='vision',
            output='screen',
            remappings=[
                ('camera', '/camera/color/image_raw/compressed')
            ],
            parameters=[
               # Correctly set as INTEGER_ARRAY
            ]
        )
    ])
