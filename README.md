# Teleoperation

ROS 2 Humble teleoperation system for collecting robot demonstration datasets compatible with **pi0.5 / openpi / LeRobot** training.

**Platform:** Ubuntu 22.04 + ROS 2 Humble
**Hardware:** Uarm master arm → UR5 follower arm + Weiss Robotics CRG 30-050 gripper

---

## Hardware

| Component | Hardware | Interface |
|---|---|---|
| Master arm | Uarm with Feetech or Zhonglin servos | USB serial |
| Follower arm | Universal Robots UR5 | Ethernet (RTDE) |
| Gripper | Weiss Robotics CRG 30-050 | USB (`/dev/ttyACM0`) |
| Cameras | Intel RealSense D4xx × 2 or 3 (auto-detected) | USB |

---

## Installation

### One-shot setup (recommended)

```bash
git clone <repo-url>
cd teleoperation
bash install.sh
```

This installs ROS 2 Humble apt packages, creates a Python venv at `.venv`, installs pip dependencies, builds the ROS 2 package, and adds your user to the `dialout` group. **Re-login after** for serial port access.

### Manual steps (if needed)

<details>
<summary>Expand manual installation</summary>

**1 — ROS 2 Humble**

```bash
sudo apt install ros-humble-desktop \
                 python3-colcon-common-extensions \
                 ros-humble-cv-bridge \
                 ros-humble-sensor-msgs \
                 ros-humble-std-msgs
```

**2 — Python venv + dependencies**

```bash
python3 -m venv teleop
source teleop/bin/activate
pip install -r requirements.txt
```

**3 — Build the ROS 2 package**

```bash
colcon build --packages-select uarm
source install/setup.bash
```

**4 — Serial port permissions**

```bash
sudo usermod -aG dialout $USER   # re-login after
```

</details>

---

## Per-Machine Configuration

These values must be updated to match each machine's hardware before running:

### Camera serial numbers — `data_collection/cam_pub.py`

```python
SERIAL_1 = "105422061000"  # exterior camera (D415)  [required]
SERIAL_2 = "352122273671"  # wrist camera    (D405)  [required]
SERIAL_3 = "104122061227"  # front camera    (D415, USB 2.1) [optional — auto-detected]
```

The front camera (`SERIAL_3`) is optional. At startup `cam_pub.py` queries pyrealsense2 for connected devices; if `SERIAL_3` is not found it runs in 2-camera mode automatically. The recorder and replay viewer adapt accordingly.

Find your serial numbers:
```bash
python3 -c "import pyrealsense2 as rs; [print(d.get_info(rs.camera_info.serial_number), d.get_info(rs.camera_info.name)) for d in rs.context().query_devices()]"
```

### UR5 and gripper constants — `uarm/scripts/UR5/servo2ur5.py` and `ur5_pub.py`

| Constant | Default | Description |
|---|---|---|
| `UR5_IP` | `10.0.0.1` | UR5 controller IP |
| `GRIPPER_PORT` | `/dev/ttyACM0` | CRG 30-050 USB port |
| `UR5_HOME_DEG` | `[0,-90,90,-90,-90,0]` | UR5 home joint angles (deg) |
| `JOINT_SCALE` | `[1,1,1,1,1,1]` | Sign correction per joint |
| `MAX_JOINT_VEL_RAD` | `1.0` | Safety velocity cap (rad/s) |
| `CONTROL_HZ` | `60` | servoJ streaming rate |
| `GRIPPER_HZ` | `10` | Gripper serial command rate |
| `GRIPPER_MAX_MM` | `30.0` | CRG 30-050 max stroke (mm) |

> Keep `UR5_IP`, `GRIPPER_PORT`, and `GRIPPER_MAX_MM` in sync between `servo2ur5.py` and `ur5_pub.py`.

---

## Quick Start

```bash
# If not using direnv, source the workspace manually (once per shell)
source install/setup.bash

# Launch all nodes
bash uarm/scripts/UR5/run_ur5_nodes.sh
```

This starts five nodes in parallel:

| # | Script | Publishes |
|---|---|---|
| 1 | `cam_pub.py` | `/cam_1`, `/cam_2` [, `/cam_3` if front camera connected] |
| 2 | `ur5_pub.py` | `/robot_state` |
| 3 | `feetech_servo_reader.py` | `/servo_angles` |
| 4 | `servo2ur5.py` | `/robot_action` |
| 5 | `episode_recorder.py` | *(writes HDF5 to disk)* |

Press `Ctrl+C` to shut down all nodes cleanly.

### Collect data

Focus the **OpenCV preview window**, then use these keys:

| Key | State | Action |
|-----|-------|--------|
| `P` | any | Set task language prompt *(once per session)* |
| `B` | WAITING | Begin recording *(requires prompt)* |
| `S` | RECORDING | Stop and save episode |
| `D` | WAITING | Delete last saved episode *(undo)* |
| `Q` | any | Quit |

The recorder auto-stops and saves when `episode_len` steps are reached.

### Advanced: override recorder settings

```bash
python3 data_collection/episode_recorder.py \
    --task default --data-dir /data/my_task --hz 20 --max-steps 300
```

Default config in `episode_recorder.py`:

```python
TASK_CONFIGS = {
    "default": {
        "dataset_dir": "openpi/dataset/ur5_dataset_YYYYMMDD",   # date auto-filled
        "episode_len": 500,
        "hz": 15,
    },
}
```

---

## Dataset Format

Episodes are saved as `episode_0.hdf5`, `episode_1.hdf5`, … in the configured dataset directory.

```
episode_N.hdf5
├── observations/
│   ├── images/
│   │   ├── exterior_image_1_left   (T, 224, 224, 3)  uint8   lzf
│   │   ├── wrist_image_left        (T, 224, 224, 3)  uint8   lzf
│   │   └── front_image_1           (T, 224, 224, 3)  uint8   lzf  [3-camera only]
│   └── qpos                        (T, 7)             float64 gzip
└── action                          (T, 7)             float64 gzip

attrs: sim, prompt, task, hz, n_steps, timestamp, num_cameras
```

`qpos` / `action` layout: `[joint_0 … joint_5 (deg), gripper (0=closed, 1=open)]`

---

## Folder Structure

```
teleoperation/
├── install.sh                         # One-shot setup for a new PC
├── .envrc                             # direnv: auto-activates ROS 2 + venv on cd
├── teleop/                            # Python venv (created by install.sh, git-ignored)
├── data_collection/                   # Robot-agnostic data tools
│   ├── episode_recorder.py            # HDF5 dataset recorder (pi0.5 format)
│   ├── cam_pub.py                     # Adaptive RealSense camera publisher (2 or 3 cams)
│   ├── replay_episode.py              # Offline HDF5 episode viewer (no ROS, 2/3-cam aware)
│   └── test_cam.py                    # Camera hardware sanity check (auto-detects all cams)
│
└── uarm/                              # ROS 2 package (ament_python)
    ├── package.xml
    ├── setup.py / setup.cfg
    ├── resource/uarm
    ├── uarm/__init__.py
    └── scripts/
        ├── UR5/
        │   ├── servo2ur5.py           # UR5 teleoperation controller (60 Hz servoJ)
        │   ├── ur5_pub.py             # UR5 state publisher → /robot_state
        │   └── run_ur5_nodes.sh       # Launch all nodes
        ├── Uarm_teleop/
        │   ├── Feetech_servo/         # Feetech servo master arm reader
        │   └── Zhonglin_servo/        # Zhonglin servo master arm reader
        └── add_permission.sh          # Grant serial port permissions
```

---

## ROS 2 Topic Graph

```
Uarm_teleop/feetech_servo_reader.py
        │
        │  /servo_angles  (Float64MultiArray, 7 floats — degrees)
        ▼
UR5/servo2ur5.py  ──────────────────────→  /robot_action  (Float64MultiArray, 7)
                                                            joints [deg ×6] + gripper [0-1]
UR5/ur5_pub.py    ──────────────────────→  /robot_state   (Float64MultiArray, 7)

data_collection/cam_pub.py              →  /cam_1  (exterior, sensor_msgs/Image)
                                        →  /cam_2  (wrist,    sensor_msgs/Image)
                                        →  /cam_3  (front,    sensor_msgs/Image)  [if connected]

data_collection/episode_recorder.py   subscribes to all topics above
                                       (auto-detects /cam_3; records 2 or 3 images per step)
```

---

<details>
<summary>ROS 1 → ROS 2 Migration Reference</summary>

| ROS 1 (Noetic) | ROS 2 (Humble) |
|---|---|
| `rospy` | `rclpy` |
| `rospy.init_node('x')` | `super().__init__('x')` in `Node` subclass |
| `rospy.Publisher(topic, T, queue_size=N)` | `self.create_publisher(T, topic, N)` |
| `rospy.Subscriber(topic, T, cb)` | `self.create_subscription(T, topic, cb, N)` |
| `rospy.Timer(rospy.Duration(dt), cb)` | `self.create_timer(dt, cb)` *(no event arg)* |
| `rospy.Rate(hz); rate.sleep()` | `time.sleep(1/hz)` + spin thread |
| `rospy.is_shutdown()` | `rclpy.ok()` |
| `rospy.spin()` | `rclpy.spin(node)` |
| `rospy.loginfo / logwarn` | `self.get_logger().info / warning` |
| `catkin` + `CMakeLists.txt` | `ament_python` + `setup.py` |
| `rosrun pkg script` | `python3 path/to/script.py` |
| `roscore` required | Not needed |

</details>
