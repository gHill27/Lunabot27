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
```

## Running the Simulation
Launch the full arena with the robot spawned in it:

```bash
ros2 launch lunabotics_sim lunabotics_arena.launch.py
```

This brings up the Artemis arena in Gazebo, spawns the robot, starts `robot_state_publisher`, and starts the ROS↔Gazebo bridge (`/cmd_vel`, `/odom`).

## Teleoperating the Robot
Once the simulation is running, drive the robot around with the keyboard using ROS 2's built-in teleop package:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

If it's not installed:
```bash
sudo apt install ros-jazzy-teleop-twist-keyboard
```

Controls (shown on-screen when the node starts):
```
   u    i    o
   j    k    l
   m    ,    .
```
* `i` / `,` — forward / backward
* `j` / `l` — turn left / right
* `u` / `o` / `m` / `.` — diagonal movement
* `k` — stop
* `q` / `z` — increase / decrease overall speed
* `w` / `x` — increase / decrease linear speed only
* `e` / `c` — increase / decrease angular speed only

This publishes `geometry_msgs/msg/Twist` messages on `/cmd_vel`, which the bridge relays into Gazebo to drive the robot's differential drive plugin. Confirm messages are getting through with:

```bash
ros2 topic echo /cmd_vel
```