import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction


def generate_launch_description():
    pkg_share = get_package_share_directory('lunabotics_sim')
    rtabmap_share = get_package_share_directory('rtabmap_launch')
    nav2_share = get_package_share_directory('nav2_bringup')

    # Switch to the raw sim depth to test whether Depth.py filtering is the problem:
    #   depth_topic:=/camera/camera/depth/image_raw
    depth_topic = DeclareLaunchArgument('depth_topic', default_value='/camera/camera/depth/image_raw')
    visual_odom = DeclareLaunchArgument('visual_odometry', default_value='false')

    # Existing launch, in sim mode
    base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'lunabotics_arena.launch.py')),
        launch_arguments={'use_sim': 'true'}.items(),
    )
    

    # RTAB-Map
    rtabmap_params_file = os.path.join(pkg_share, 'config', 'rtabmap_params.yaml')

    rtabmap = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        os.path.join(rtabmap_share, 'launch', 'rtabmap.launch.py')),
    launch_arguments={
        'frame_id': 'base_link',
        'rgb_topic': '/camera/camera/color/image_raw',
        'camera_info_topic': '/camera/camera/color/camera_info',
        'depth_topic': LaunchConfiguration('depth_topic'),
        'subscribe_depth': 'true',
        'approx_sync': 'true',
        'visual_odometry': LaunchConfiguration('visual_odometry'),
        'odom_topic': '/odom',
        'use_sim_time': 'true',
        'qos': '2',
        'cfg': os.path.join(pkg_share, 'config', 'rtabmap.ini'),
        'args': "--delete_db_on_start",                          
        'rtabmap_viz': 'true',
        'rviz': 'false',
        'namespace': '',
        'wait_for_transform': '0.5',
    }.items(),
)

    depth_to_cloud = Node(
        package='rtabmap_util',
        executable='point_cloud_xyz',
        name='point_cloud_xyz',
        output='screen',
        parameters=[{
            'use_sim_time':True,
            'decimation': 4,       # Downsamples 640x480 to 160x120 for fast Nav2 raycasting
            'max_depth': 4.0,      # Clip noisy far points
            'voxel_size': 0.05,    # 5cm voxel grid filter
            'approx_sync': True,
        }],
        remappings=[
            ('depth/image', '/camera/camera/depth/image_raw'),
            ('depth/camera_info', '/camera/camera/color/camera_info'),
            ('cloud', '/camera/depth_registered/points')
        ]
    )


    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_share, 'launch', 'navigation_launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'params_file': os.path.join(pkg_share, 'config', 'nav2_params.yaml'),
        }.items(),
    )
    startup = Node(
        package='lunabotics_sim',
        executable='startup.py',
        name='startupNode',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # RViz2 pre-loaded with the Nav2 default panel
    rviz_config_file = os.path.join(nav2_share, 'rviz', 'nav2_default_view.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )
    # The bridge starts at 6 s, so wait for it and the camera topics
    delayed_rtabmap = TimerAction(period=10.0, actions=[rtabmap, depth_to_cloud])
    delayed_nav2 = TimerAction(period=12.0, actions=[nav2, rviz])
    delayed_startup = TimerAction(period=15.0, actions=[startup])

    # fastdds_profile = SetEnvironmentVariable(
    #     name='FASTRTPS_DEFAULT_PROFILES_FILE',
    #     value=os.path.join(pkg_share, 'config', 'fastdds_shm.xml')
    # )
    # fastdds_qos = SetEnvironmentVariable(
    #     name='RMW_FASTRTPS_USE_QOS_FROM_XML',
    #     value='1'
    # )
    #TODO: for implementatoin with wifi fix the dds profile! all local currnetly!!!!!

    return LaunchDescription([
        depth_topic,
        visual_odom,
        base_launch,
        delayed_rtabmap,
        delayed_nav2,
        delayed_startup
        ])