# Teleoperation

ROS Noetic-based teleoperation system for collecting robot demonstration datasets compatible with **pi0.5 / openpi / LeRobot** training.

**Hardware:** Uarm master arm → UR5 follower arm + Weiss Robotics CRG 30-050 gripper.

---

## Folder Structure

```
teleoperation/
├── data_collection/                   # Robot-agnostic data tools
│   ├── episode_recorder.py            # HDF5 dataset recorder (pi0.5 format)
│   ├── cam_pub.py                     # Dual RealSense camera ROS publisher
│   ├── play_episode.py                # Visualise a recorded episode
│   └── test_cam.py                    # Quick single-camera sanity check
│
└── uarm/                              # Uarm master arm + UR5 follower arm
    ├── CMakeLists.txt / package.xml
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

## ROS Topic Graph

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

data_collection/episode_recorder.py   subscribes to all four topics above
```

---

## Hardware

| Component | Hardware | Interface |
|---|---|---|
| Master arm | Uarm with Feetech or Zhonglin servos | USB serial |
| Follower arm | Universal Robots UR5 | Ethernet (RTDE) |
| Gripper | Weiss Robotics CRG 30-050 | USB (`/dev/ttyACM0`) |
| Cameras | Intel RealSense D4xx × 2 | USB |

---

## Software Requirements

```bash
# ROS Noetic
sudo apt install ros-noetic-desktop-full python3-rospy python3-cv-bridge

# Python dependencies
pip install pyrealsense2 opencv-python h5py numpy ur-rtde pyserial
```

Grant serial port access (re-login after):
```bash
bash uarm/scripts/add_permission.sh
# or manually:
sudo usermod -aG dialout $USER
```

---

## Quick Start

### 1 — Launch the pipeline

```bash
bash uarm/scripts/UR5/run_ur5_nodes.sh
```

This starts five nodes in parallel:

| # | Script | Publishes |
|---|---|---|
| 1 | `cam_pub.py` | `/cam_1`, `/cam_2` |
| 2 | `ur5_pub.py` | `/robot_state` |
| 3 | `feetech_servo_reader.py` | `/servo_angles` |
| 4 | `servo2ur5.py` | `/robot_action` |
| 5 | `episode_recorder.py` | *(writes HDF5 to disk)* |

Press `Ctrl+C` to shut down all nodes cleanly.

### 2 — Collect data

Focus the **OpenCV preview window**, then use these keys:

| Key | State | Action |
|-----|-------|--------|
| `P` | any | Set task language prompt *(once per session)* |
| `B` | WAITING | Begin recording *(requires prompt)* |
| `S` | RECORDING | Stop and save episode |
| `D` | WAITING | Delete last saved episode *(undo)* |
| `Q` | any | Quit |

The recorder auto-stops and saves when `episode_len` steps are reached.

---

## Dataset Format

Episodes are saved as `episode_0.hdf5`, `episode_1.hdf5`, … in the configured dataset directory.

```
episode_N.hdf5
├── observations/
│   ├── images/
│   │   ├── exterior_image_1_left   (T, 224, 224, 3)  uint8   lzf
│   │   └── wrist_image_left        (T, 224, 224, 3)  uint8   lzf
│   └── qpos                        (T, 7)             float64 gzip
└── action                          (T, 7)             float64 gzip

attrs: sim, prompt, task, hz, n_steps, timestamp
```

`qpos` / `action` layout: `[joint_0 … joint_5 (deg), gripper (0=closed, 1=open)]`

---

## Configuration

### episode_recorder.py — `TASK_CONFIGS`

Edit the `TASK_CONFIGS` dict at the top of the file:

```python
TASK_CONFIGS = {
    "default": {
        "dataset_dir": "~/pi05_dataset/default",
        "episode_len": 500,   # max steps before auto-save
        "hz": 15,             # recording frequency
    },
}
```

Override at runtime:

```bash
python3 data_collection/episode_recorder.py \
    --task default \
    --data-dir /data/my_task \
    --hz 20 \
    --max-steps 300
```

### cam_pub.py — Camera serial numbers

```python
config_1.enable_device('338622073582')  # exterior camera
config_2.enable_device('148522073685')  # wrist camera
```

Find your serial numbers:
```bash
rs-enumerate-devices | grep Serial
```

### servo2ur5.py and ur5_pub.py — Key constants

| Constant | Default | Description |
|---|---|---|
| `UR5_IP` | `192.168.1.100` | UR5 controller IP |
| `GRIPPER_PORT` | `/dev/ttyACM0` | CRG 30-050 USB port |
| `UR5_HOME_DEG` | `[0,-90,90,-90,-90,0]` | UR5 home joint angles (deg) |
| `JOINT_SCALE` | `[1,1,1,1,1,1]` | Sign correction per joint |
| `MAX_JOINT_VEL_RAD` | `1.0` | Safety velocity cap (rad/s) |
| `CONTROL_HZ` | `60` | servoJ streaming rate |
| `GRIPPER_HZ` | `10` | Gripper serial command rate |
| `GRIPPER_MAX_MM` | `30.0` | CRG 30-050 max stroke (mm) |

> Keep `UR5_IP`, `GRIPPER_PORT`, and `GRIPPER_MAX_MM` in sync between `servo2ur5.py` and `ur5_pub.py`.
