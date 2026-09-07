# Lunabotics Sim 2027

This repository contains the simulation environment for our robot for the 2027 NASA lunabotics challenge. The simulation is built for **ROS 2 Jazzy** and **Gazebo Harmonic**.

## Prerequisites
Ensure you have the following installed on your system:
* Ubuntu 24.04 (Noble)
* ROS 2 Jazzy
* Gazebo Harmonic

## Building the Workspace
Clone this repository into your ROS 2 workspace (e.g., `~/ros_ws/src`), then build the package using `colcon`:

```bash
cd ~/ros_ws
colcon build --symlink-install --packages-select lunabotics_sim
source install/setup.bash
