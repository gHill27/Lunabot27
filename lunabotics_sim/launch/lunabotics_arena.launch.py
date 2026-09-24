import os
import xacro
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim = LaunchConfiguration('use_sim')
    use_sim_arg = DeclareLaunchArgument(
        'use_sim',
        default_value='false',
        description='Set to true to launch Gazebo simulation instead of real hardware'
    )
    pkg_path = get_package_share_directory('lunabotics_sim')

    # Bring up the default competition arena (artemis_arena) via the
    # existing launch file — same GZ_SIM_RESOURCE_PATH setup as the
    # standalone arena launch files.
    arena = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_path, 'launch', 'artemis_arena.launch.py')
        ),
        condition=IfCondition(use_sim),
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
        condition=IfCondition(use_sim),
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
            '-x', '1.0', '-y', '-1.0', '-z', '1.0',
        ],
        output='screen',
        condition=IfCondition(use_sim)
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
        condition=IfCondition(use_sim)
    )

    fiducial_node = Node(
        package='lunabotics_sim',
        executable='fiducial.py',
        name='fiducial_node',
        output='screen',
    )
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'
            ])
        ),
        launch_arguments={
            'enable_color': 'true',
            'enable_depth': 'true',
            'rgb_camera.color_profile': '640x480x30',
            'depth_module.depth_profile': '640x480x30',
        }.items(),
        condition=UnlessCondition(use_sim)  # only run the real camera when NOT in sim
    )

    # Give the arena a few seconds to come up (Sun model fetch from Fuel,
    # scene setup) before spawning the robot.
    delayed_spawn = TimerAction(period=5.0, actions=[spawn_entity])

    # Bridge only needs to exist once the robot (and its plugin) is
    # actually spawned.
    delayed_bridge = TimerAction(period=6.0, actions=[ros_gz_bridge])

    return LaunchDescription([
        arena,
        realsense_launch,
        robot_state_publisher,
        delayed_spawn,
        delayed_bridge,
        fiducial_node,
    ])
