#!/usr/bin/env python3
"""
UR5 + Weiss Robotics CRG 30-050 Teleoperation Node

Subscribes: /servo_angles  (Float64MultiArray, 7 floats in degrees relative to home,
                             published by Uarm_teleop master arm)
Publishes:  /robot_action  (Float64MultiArray, 7 floats:
                             joints 0-5 in degrees + gripper normalised 0-1)

Hardware:
    - UR5 connected via Ethernet (RTDE, port 30004)
    - Weiss Robotics CRG 30-050 connected via USB cable
      (appears as /dev/ttyACM0 on Linux; run `ls /dev/ttyACM*` to confirm)

Dependencies:
    pip install ur-rtde pyserial
"""

import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

try:
    import rtde_control
    import rtde_receive
except ImportError as err:
    raise ImportError("ur_rtde not installed. Run: pip install ur-rtde") from err

try:
    import serial
except ImportError as err:
    raise ImportError("pyserial not installed. Run: pip install pyserial") from err

# ══════════════════════════════ Configuration ══════════════════════════════

UR5_IP = "192.168.1.100"  # ← change to your UR5 controller IP

# CRG 30-050 USB serial port (Linux: /dev/ttyACM0, may vary)
GRIPPER_PORT = "/dev/ttyACM0"
GRIPPER_BAUDRATE = 115200

# CRG 30-050: 30 mm total stroke, 28 N max force
GRIPPER_MAX_MM = 30.0

# Gripper opens/closes as a binary device (PDOUT=[07,00] open, PDOUT=[03,00] close).
# Threshold: if commanded width > this value → open; otherwise → close.
GRIPPER_OPEN_THRESHOLD_MM = 5.0

# UR5 home/initial joint position (degrees).
# [shoulder_pan, shoulder_lift, elbow, wrist1, wrist2, wrist3]
UR5_HOME_DEG = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]

# Per-joint scale factors (sign correction) from master arm to UR5.
JOINT_SCALE = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# servoJ parameters
SERVO_J_TIME = 0.016  # seconds per command step (≈ 60 Hz)
SERVO_J_LOOKAHEAD = 0.08
SERVO_J_GAIN = 300

# UR5 joint velocity limit (rad/s) for safety clipping (~57 deg/s)
MAX_JOINT_VEL_RAD = 1.0

# Master-arm gripper channel (index 6): degree range → fully closed / open.
GRIPPER_SERVO_CLOSED_DEG = -5.0
GRIPPER_SERVO_OPEN_DEG = 35.0

CONTROL_HZ = 60  # main joint control loop rate
GRIPPER_HZ = 10  # gripper command rate

# ══════════════════════════════ Weiss CRG 30-050 ══════════════════════════════


class WeissCRGGripper:
    """
    USB serial driver for Weiss Robotics CRG 30-050 (firmware 1.0.7).

    Protocol: PDOUT/SETPARAM ASCII over USB CDC-ACM (DC-IOLINK).
    The gripper continuously pushes @PDIN=[B0,B1,B2,B3],N messages.

    Key commands:
        PDOUT=[07,00]              – open fully (reference/home motion, ~1 s)
        PDOUT=[03,00]              – close fully (stops at force limit / jaw contact)
        PDOUT=[02,00]              – grip (close until force detected)
        PDOUT=[00,00]              – stop / deactivate
    @PDIN position encoding: jaw_gap_mm = (B0<<8|B1 - closed_pdin) / PDIN_PER_MM
    Empirical constants (measured): PDIN_PER_MM ≈ 163.17, closed_pdin ≈ 150 (varies).

    Motion model (binary open/close):
        PDOUT=[07,00]   open  → PDIN ≈ 5055 (30 mm), motion takes ~1 s
        PDOUT=[03,00]   close → PDIN ≈ 130–160 (0 mm), stops at force limit
        PDOUT=[02,00]   grip  → closes until force detected (use for active gripping)
    SETPARAM(96) does NOT give proportional position; gripper is binary open/close.
    """

    PDIN_PER_MM  = 163.17       # PDIN encoder units per mm of jaw gap (measured)

    def __init__(self, port: str = GRIPPER_PORT, baudrate: int = GRIPPER_BAUDRATE, logger=None):
        self._lock = threading.Lock()
        self._logger = logger
        self._ser = serial.Serial(port=port, baudrate=baudrate, timeout=2.0)
        self._position_mm = 0.0
        self._closed_pdin = 150   # updated by home(); encoder count at fully closed
        self._log_info(f"Opened {port} @ {baudrate} baud")

    def _log_info(self, msg: str):
        if self._logger:
            self._logger.info(f"[WeissCRG] {msg}")

    def _log_warn(self, msg: str):
        if self._logger:
            self._logger.warning(f"[WeissCRG] {msg}")

    def _parse_pdin(self, line: str):
        """Update cached position from @PDIN=[B0,B1,B2,B3],N push message."""
        try:
            data = line[7:].split("]")[0].split(",")
            pdin_raw = (int(data[0], 16) << 8) | int(data[1], 16)
            self._position_mm = max(0.0, (pdin_raw - self._closed_pdin) / self.PDIN_PER_MM)
        except Exception:
            pass

    def _send(self, cmd: str) -> str:
        """Send command; skip @PDIN push lines; return first command response."""
        with self._lock:
            try:
                self._ser.reset_input_buffer()
                self._ser.write((cmd + "\n").encode("ascii"))
                for _ in range(30):
                    resp = self._ser.readline().decode("ascii", errors="ignore").strip()
                    if resp.startswith("@PDIN"):
                        self._parse_pdin(resp)
                        continue
                    return resp
                return ""
            except Exception as e:
                self._log_warn(f"Serial error on '{cmd}': {e}")
                return ""

    def get_width(self) -> float:
        """Read a fresh @PDIN position message from the gripper."""
        with self._lock:
            try:
                saved = self._ser.timeout
                self._ser.timeout = 0.3
                for _ in range(20):
                    line = self._ser.readline().decode("ascii", errors="ignore").strip()
                    if line.startswith("@PDIN"):
                        self._parse_pdin(line)
                        break
                self._ser.timeout = saved
            except Exception:
                pass
        return self._position_mm

    def _wait_motion(self, timeout: float = 5.0) -> float:
        """Read live @PDIN until motion completes. Returns final PDIN value."""
        start_pdin = None
        prev = None
        stable = 0
        moved = False
        t0 = time.monotonic()
        with self._lock:
            saved = self._ser.timeout
            self._ser.timeout = 0.1
        try:
            while time.monotonic() - t0 < timeout:
                with self._lock:
                    line = self._ser.readline().decode("ascii", errors="ignore").strip()
                if not line.startswith("@PDIN"):
                    continue
                try:
                    data = line[7:].split("]")[0].split(",")
                    v = (int(data[0], 16) << 8) | int(data[1], 16)
                except Exception:
                    continue
                if start_pdin is None:
                    start_pdin = v
                if not moved and abs(v - start_pdin) >= 200:
                    moved = True
                if moved:
                    if prev is not None and abs(v - prev) <= 5:
                        stable += 1
                        if stable >= 15:
                            self._position_mm = max(0.0, (v - self._closed_pdin) / self.PDIN_PER_MM)
                            return v
                    else:
                        stable = 0
                prev = v
        finally:
            with self._lock:
                self._ser.timeout = saved
        return prev or 0

    def home(self) -> bool:
        """Close fully to calibrate closed_pdin, then open to ready position."""
        self._log_info("Calibrating closed position (PDOUT=[03,00])...")
        self._send("PDOUT=[00,00]")
        time.sleep(0.1)
        self._send("PDOUT=[03,00]")
        closed = self._wait_motion(5.0)
        self._closed_pdin = int(closed) if closed else 150
        self._position_mm = 0.0
        self._log_info(f"closed_pdin={self._closed_pdin}, opening gripper...")
        self._send("PDOUT=[00,00]")
        time.sleep(0.1)
        self._send("PDOUT=[07,00]")
        self._wait_motion(5.0)
        self._log_info("Homing done, gripper open and ready.")
        return True

    def move_to_pos(self, width_mm: float) -> bool:
        """Open (width_mm > threshold) or close the gripper."""
        if width_mm > GRIPPER_OPEN_THRESHOLD_MM:
            self._send("PDOUT=[00,00]")  # release hold before opening
            self._send("PDOUT=[07,00]")
        else:
            self._send("PDOUT=[00,00]")  # release reference-hold before closing
            self._send("PDOUT=[03,00]")
        return True

    def close(self):
        self._send("PDOUT=[00,00]")
        if self._ser and self._ser.is_open:
            self._ser.close()
            self._log_info("Serial port closed")


# ══════════════════════════════ UR5 Teleop Node ══════════════════════════════


class UR5TeleopNode(Node):
    """
    Reads Uarm master-arm joint angles, maps them to the UR5 frame,
    streams joint commands via RTDE servoJ, and drives the CRG 30-050 gripper.
    """

    def __init__(self):
        super().__init__("ur5_teleop_node")

        # ── UR5 RTDE ─────────────────────────────────────────────
        self.get_logger().info(f"[UR5Teleop] Connecting to UR5 at {UR5_IP}...")
        self.rtde_c = rtde_control.RTDEControlInterface(UR5_IP)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(UR5_IP)
        self.get_logger().info("[UR5Teleop] UR5 RTDE connected.")

        # ── Weiss CRG 30-050 ──────────────────────────────────────
        self.get_logger().info(f"[UR5Teleop] Opening gripper on {GRIPPER_PORT}...")
        self.gripper = WeissCRGGripper(GRIPPER_PORT, GRIPPER_BAUDRATE, logger=self.get_logger())
        self.gripper.home()

        # ── UR5 home position ─────────────────────────────────────
        self.home_rad = np.radians(UR5_HOME_DEG)
        self.get_logger().info(f"[UR5Teleop] Moving UR5 to home: {UR5_HOME_DEG} deg")
        self.rtde_c.moveJ(self.home_rad.tolist(), speed=0.5, acceleration=0.5)
        self.get_logger().info("[UR5Teleop] UR5 at home position.")

        # ── shared command state ──────────────────────────────────
        self._lock = threading.Lock()
        self._cmd_joints_rad = self.home_rad.copy()
        self._cmd_gripper_mm = 0.0
        self._last_cmd_rad = self.home_rad.copy()
        self._servo_warned = False

        # ── ROS 2 interfaces ──────────────────────────────────────
        self._pub = self.create_publisher(Float64MultiArray, "/robot_action", 10)
        self.create_subscription(Float64MultiArray, "/servo_angles", self._servo_callback, 10)

        # ── gripper command thread ────────────────────────────────
        self._gripper_thread = threading.Thread(target=self._gripper_loop, daemon=True)
        self._gripper_thread.start()

        self.get_logger().info(
            f"[UR5Teleop] Ready — streaming servoJ at {CONTROL_HZ} Hz, " f"gripper at {GRIPPER_HZ} Hz."
        )

    # ── angle mapping ─────────────────────────────────────────────────────────

    def _map_joints(self, servo_deg: np.ndarray) -> np.ndarray:
        delta_rad = np.deg2rad(servo_deg) * np.array(JOINT_SCALE)
        return self.home_rad + delta_rad

    def _map_gripper(self, servo_deg: float) -> float:
        norm = (servo_deg - GRIPPER_SERVO_CLOSED_DEG) / max(1e-6, GRIPPER_SERVO_OPEN_DEG - GRIPPER_SERVO_CLOSED_DEG)
        return float(np.clip(norm, 0.0, 1.0)) * GRIPPER_MAX_MM

    def _limit_joint_vel(self, target: np.ndarray, dt: float) -> np.ndarray:
        max_step = MAX_JOINT_VEL_RAD * dt
        delta = target - self._last_cmd_rad
        return self._last_cmd_rad + np.clip(delta, -max_step, max_step)

    # ── subscription callback ─────────────────────────────────────────────────

    def _servo_callback(self, msg: Float64MultiArray):
        data = np.asarray(msg.data, dtype=np.float64)
        if data.size < 7:
            if not self._servo_warned:
                self.get_logger().warning("[UR5Teleop] /servo_angles needs 7 values (6 joints + gripper)")
                self._servo_warned = True
            return
        joints_rad = self._map_joints(data[:6])
        gripper_mm = self._map_gripper(float(data[6]))
        with self._lock:
            self._cmd_joints_rad = joints_rad
            self._cmd_gripper_mm = gripper_mm

    # ── gripper thread ────────────────────────────────────────────────────────

    def _gripper_loop(self):
        dt = 1.0 / GRIPPER_HZ
        last_open = True  # gripper starts open after home()
        while rclpy.ok():
            with self._lock:
                target_mm = self._cmd_gripper_mm
            want_open = target_mm > GRIPPER_OPEN_THRESHOLD_MM
            if want_open != last_open:
                self.gripper.move_to_pos(target_mm)
                last_open = want_open
                self.get_logger().info(
                    f"[UR5Teleop] Gripper {'opening' if want_open else 'closing'}"
                )
            time.sleep(dt)

    # ── main control loop ─────────────────────────────────────────────────────

    def run(self):
        dt = 1.0 / CONTROL_HZ
        while rclpy.ok():
            t0 = time.monotonic()

            with self._lock:
                target_rad = self._cmd_joints_rad.copy()
                gripper_mm = self._cmd_gripper_mm

            cmd_rad = self._limit_joint_vel(target_rad, dt)
            self._last_cmd_rad = cmd_rad.copy()

            try:
                self.rtde_c.servoJ(
                    cmd_rad.tolist(),
                    velocity=0,
                    acceleration=0,
                    time=SERVO_J_TIME,
                    lookahead_time=SERVO_J_LOOKAHEAD,
                    gain=SERVO_J_GAIN,
                )
            except Exception as e:
                self.get_logger().warning(f"[UR5Teleop] servoJ error: {e}")

            gripper_norm = float(np.clip(gripper_mm / GRIPPER_MAX_MM, 0.0, 1.0))
            action_data = [*np.degrees(cmd_rad).tolist(), gripper_norm]
            msg = Float64MultiArray()
            msg.data = action_data
            self._pub.publish(msg)
            self.get_logger().debug(f"[UR5Teleop] action: {[f'{v:.2f}' for v in action_data]}")

            elapsed = time.monotonic() - t0
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        # Clean shutdown
        self.rtde_c.servoStop()
        self.get_logger().info("[UR5Teleop] servoJ stopped.")
        self.gripper.close()


# ══════════════════════════════ Entry point ══════════════════════════════


def main():
    rclpy.init()
    node = UR5TeleopNode()

    # Spin subscriptions in a background thread; run() holds the control loop.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
