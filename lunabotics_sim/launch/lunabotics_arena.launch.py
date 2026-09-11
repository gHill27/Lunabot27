import os

import xacro
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_path = get_package_share_directory('lunabotics_sim')

    # Bring up the default competition arena (artemis_arena) via the
    # existing launch file — same GZ_SIM_RESOURCE_PATH setup as the
    # standalone arena launch files.
    arena = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_path, 'launch', 'artemis_arena.launch.py')
        )
    )

    # Process the xacro into a robot_description string.
    xacro_file = os.path.join(pkg_path, 'urdf', 'lunabot.urdf.xacro')
    robot_description_config = xacro.process_file(xacro_file)
    robot_description = {'robot_description': robot_description_config.toxml()}

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )

    # Spawn a clear spot in artemis_arena, away from the rock clusters —
    # checked against the rock coordinates in worlds/artemis_arena.sdf and
    # against lunar_surface_a's floor bounds (8.14m x 9.14m).
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', '/robot_description',
            '-name', 'lunabot',
            '-x', '2.5', '-y', '-1.5', '-z', '1.0',
        ],
        output='screen',
    )

    # Bridge /cmd_vel (ROS Twist) <-> Gazebo Transport, and /odom
    # (Gazebo -> ROS) so the DiffDrive system plugin on the robot can
    # actually be driven from ROS 2 nodes (e.g. teleop_keyboard.py).
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            # RealSense-style RGB-D camera, mounted top-middle of chassis
            '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
        ],
        output='screen',
    )

    # Give the arena a few seconds to come up (Sun model fetch from Fuel,
    # scene setup) before spawning the robot.
    delayed_spawn = TimerAction(period=5.0, actions=[spawn_entity])

    # Bridge only needs to exist once the robot (and its plugin) is
    # actually spawned.
    delayed_bridge = TimerAction(period=6.0, actions=[ros_gz_bridge])

    return LaunchDescription([
        arena,
        robot_state_publisher,
        delayed_spawn,
        delayed_bridge,
    ])
