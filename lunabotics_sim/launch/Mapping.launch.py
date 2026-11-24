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
    realsense_sync = Node(
        package='rtabmap_sync',
        executable='rgbd_sync',
        name='realsense_rgbd_sync',
        output='screen',
        parameters=[{
            'approx_sync': True,
            'approx_sync_max_interval': 0.02,
            'use_sim_time': True,
        }],
        remappings=[
            ('rgb/image', '/camera/camera/color/image_raw'),
            ('depth/image', '/camera/camera/depth/image_raw'),
            ('rgb/camera_info', '/camera/camera/color/camera_info'),
            ('rgbd_image', '/camera/camera/rgbd_image'),
        ],
    )

    orbbec_sync = Node(
        package='rtabmap_sync',
        executable='rgbd_sync',
        name='orbbec_rgbd_sync',
        output='screen',
        parameters=[{
            'approx_sync': True,
            'approx_sync_max_interval': 0.02,
            'use_sim_time': True,
        }],
        remappings=[
            ('rgb/image', '/orbbec_camera/image'),
            ('depth/image', '/orbbec_camera/depth_image'),
            ('rgb/camera_info', '/orbbec_camera/camera_info'),
            ('rgbd_image', '/orbbec_camera/rgbd_image'),
        ],
    )
    multi_camera_sync = Node(
        package='rtabmap_sync',
        executable='rgbdx_sync',
        name='rgbdx_sync',
        output='screen',
        parameters=[{
            'rgbd_cameras': 2,
            'approx_sync': True,
            'approx_sync_max_interval': 0.05,
            'use_sim_time': True,
        }],
        remappings=[
            ('rgbd_image0', '/camera/camera/rgbd_image'),
            ('rgbd_image1', '/orbbec_camera/rgbd_image'),
            ('rgbd_images', '/rgbd_images'),
        ],
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[os.path.join(pkg_share, 'config', 'ekf.yaml')],
    )

    # RTAB-Map
    rtabmap_common_params = [
        os.path.join(pkg_share, 'config', 'rtab_params.yaml'),
        {
            'frame_id': 'base_link',
            'subscribe_depth': False,
            'subscribe_rgb': False,
            'subscribe_rgbd': True,
            'rgbd_cameras': 0,          # 0 = subscribe to multi-camera RGBDImages array
            'subscribe_scan': True,
            'subscribe_odom': True,
            'approx_sync': True,
            'qos': 2,
            'wait_for_transform': 0.5,
            'use_sim_time': True,
            'database_path': '',
        },
    ]

    rtabmap_remappings = [
        ('rgbd_images', '/rgbd_images'),
        ('scan', '/scan'),
        ('odom', '/odom'),
    ]

    rtabmap = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        output='screen',
        parameters=rtabmap_common_params,
        remappings=rtabmap_remappings,
        arguments=['--delete_db_on_start'],
    )

    rtabmap_viz = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        name='rtabmap_viz',
        output='screen',
        parameters=rtabmap_common_params,
        remappings=rtabmap_remappings,
    )

    obstacles = Node(
        package='rtabmap_util',
        executable='obstacles_detection',
        name='obstacles_detection',
        output='screen',
        parameters=[{
            'frame_id': 'base_link',
            'map_frame_id': 'map',
            'use_sim_time': True,            # matches your rtabmap include
            'qos': 2,                        # int here, not a string
            'wait_for_transform': 0.5,
            'Grid/RangeMax': '4.0',
            'Grid/CellSize': '0.05',
            'Grid/NormalsSegmentation': 'true',
            'Grid/MaxGroundAngle': '45',
            'Grid/MaxGroundHeight': '0.05',
            'Grid/MaxObstacleHeight': '1.5',
            'Grid/MinClusterSize': '20',
            'Grid/FlatObstacleDetected': 'true',
        }],
        remappings=[
            ('cloud', '/camera/camera/depth/color/points'),   # your PointCloud2 topic
        ],
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
    delayed_rtabmap = TimerAction(period=10.0, actions=[rtabmap, rtabmap_viz, depth_to_cloud, obstacles])
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
        ekf_node,
        realsense_sync,
        orbbec_sync,
        multi_camera_sync,
        # delayed_rtabmap,
        # delayed_nav2,
        # delayed_startup
        ])