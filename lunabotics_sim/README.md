# lunabotics_sim

ROS2 Jazzy + Gazebo Harmonic (`gz-sim` / `ros_gz_sim`) simulation package for
WPI Lunabotics. Combines three competition-arena worlds with obstacles and
AprilTags, plus a placeholder robot description, in a single self-contained
package.

## Contents

- **Arenas** (`worlds/`, `models/`): `artemis_arena` (default/primary
  competition arena), `ucf_arena`, and `arena_b`, each with rocks and
  AprilTags placed as fixed-pose includes. `artemis_arena` and `ucf_arena`
  pull their ground-plane floor mesh via a nested `<include>` inside their
  respective `model.sdf` (`lunar_surface_a` and `lunar_surface_c`); `arena_b`
  is self-contained (single STL, no separate floor model).
- **Robot** (`urdf/lunabot.urdf.xacro`): a placeholder box chassis (0.6m x
  0.5m x 0.2m, 15kg) on 4 free-rolling cylindrical wheels (0.12m radius,
  1.5kg each). No drivetrain or control plugin yet — wheels are `continuous`
  joints that roll under physics but nothing commands them. A
  `gz-sim-joint-state-publisher-system` plugin publishes `/joint_states`.
  Swap in real chassis geometry once the mechanical design is locked.

## Launch files

| File | What it does |
|---|---|
| `launch/spawn_lunabot.launch.py` | Arena (`artemis_arena`) + robot spawn together — **use this one** for most work. |
| `launch/artemis_arena.launch.py` | `artemis_arena` world only. |
| `launch/ucf_arena.launch.py` | `ucf_arena` world only. |
| `launch/arena_b.launch.py` | `arena_b` world only. |

The three arena-only launch files are kept separate rather than
parameterized into one launch file with a `world` argument, by preference.

## Build & run

```bash
colcon build --symlink-install --packages-select lunabotics_sim
source install/setup.bash
ros2 launch lunabotics_sim spawn_lunabot.launch.py
```

To bring up just an arena without the robot:

```bash
ros2 launch lunabotics_sim artemis_arena.launch.py
# or ucf_arena.launch.py / arena_b.launch.py
```

## Requirements

- ROS2 Jazzy
- Gazebo Harmonic (`gz-harmonic`) + `ros_gz_sim`
- `xacro`, `robot_state_publisher` (both declared as `exec_depend` in
  `package.xml`)
- Internet access at launch time — the `Sun` light in all three worlds is
  pulled live from `https://fuel.gazebosim.org`. If you need to run
  offline, set up a local Fuel cache beforehand.

## Not done yet

- No drivetrain/control (`cmd_vel` interface via diff-drive or skid-steer
  plugin), no `ros_gz_bridge` wiring, no teleop.
- No sensors (IMU, camera, LIDAR).
- No navigation or excavation-mechanism stack.
- Placeholder robot geometry — not real Lunabotics dimensions.

This package only proves the arena + robot spawn pipeline works; it's the
foundation for the rest of the autonomy/mechanism stack, not a complete
robot.
