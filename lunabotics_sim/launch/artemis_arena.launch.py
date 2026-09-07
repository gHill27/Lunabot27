import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    ros_gz_sim_dir = get_package_share_directory('ros_gz_sim')
    lunabotics_sim_dir = get_package_share_directory('lunabotics_sim')
    
    # Define paths to your world and models
    world_path = os.path.join(lunabotics_sim_dir, 'worlds', 'artemis_arena.sdf')
    models_dir = os.path.join(lunabotics_sim_dir, 'models')
    
    # Export the models directory to Gazebo's resource path
    set_model_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=models_dir
    )
    
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_dir, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_path}'}.items()
    )
    
    return LaunchDescription([set_model_path, gz_sim])