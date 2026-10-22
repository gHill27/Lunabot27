import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_share = get_package_share_directory('lunabotics_sim')
    rtabmap_share = get_package_share_directory('rtabmap_launch')

    # Switch to the raw sim depth to test whether Depth.py filtering is the problem:
    #   depth_topic:=/camera/camera/depth/image_raw
    depth_topic = DeclareLaunchArgument('depth_topic', default_value='/depth_filtered')
    visual_odom = DeclareLaunchArgument('visual_odometry', default_value='false')

    # 1. Your existing launch, in sim mode
    base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'lunabotics_arena.launch.py')),
        launch_arguments={'use_sim': 'true'}.items(),
    )

    # 2. RTAB-Map
    rtabmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rtabmap_share, 'launch', 'rtabmap.launch.py')),
        launch_arguments={
            'frame_id': 'base_link',    # verify against your URDF
            'rgb_topic': '/camera/camera/color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'depth_topic': LaunchConfiguration('depth_topic'),
            'subscribe_depth': 'true',
            'approx_sync': 'true',
            'visual_odometry': LaunchConfiguration('visual_odometry'),
            'odom_topic': '/odom',
            'use_sim_time': 'true',
            'qos': '2',                 # best effort; also connects to reliable publishers
            'args': '-d',               # fresh map database each run
            'rtabmap_viz': 'true',
            'rviz': 'false',            # your base launch already starts RViz
        }.items(),
    )

    # The bridge starts at 6 s, so wait for it and the camera topics
    delayed_rtabmap = TimerAction(period=10.0, actions=[rtabmap])

    return LaunchDescription([depth_topic, visual_odom, base_launch, delayed_rtabmap])