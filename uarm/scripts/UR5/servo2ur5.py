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

UR5_IP = "10.0.0.1"  # ← change to your UR5 controller IP

# CRG 30-050 USB serial port (Linux: /dev/ttyACM0, may vary)
GRIPPER_PORT = "/dev/ttyACM0"
GRIPPER_BAUDRATE = 9600

# CRG 30-050: 30 mm total stroke, 28 N max force
GRIPPER_MAX_MM = 30.0

# Gripper opens/closes as a binary device (PDOUT=[02,00] open, PDOUT=[03,00] close).
# Threshold: if commanded width > this value → open; otherwise → close.
GRIPPER_OPEN_THRESHOLD_MM = 15.0  # 50% of GRIPPER_MAX_MM; tune based on master arm range

# UR5 home/initial joint position (degrees).
# [shoulder_pan, shoulder_lift, elbow, wrist1, wrist2, wrist3]
UR5_HOME_DEG = [45.0, -100.0, -120.0, 15.0, -270.0, 0.0]

# Reorder master-arm servo indices to UR5 joint indices.
# JOINT_MAP[i] = which servo channel drives UR5 joint i.
# Swap master motor 4 → UR5 joint 5, master motor 5 → UR5 joint 4.
JOINT_MAP = [0, 1, 2, 3, 5, 4]

# Per-joint scale factors (sign correction) applied after reordering.
JOINT_SCALE = [0.4, -0.3, -0.5, 0.8, 0.8, 0.3]

# servoJ parameters — balance between responsiveness and smoothness
# lookahead_time: 0.03–0.2 s  — higher = smoother motion, more lag
# gain:           100–2000    — lower = softer/less jerky, less tight tracking
SERVO_J_TIME = 0.016       # s per step (must match 1/CONTROL_HZ)
SERVO_J_LOOKAHEAD = 0.15   # s look-ahead (raised from 0.06 → smoother interpolation)
SERVO_J_GAIN = 200          # stiffness (lowered from 500 → less snapping)

# UR5 joint velocity limit (rad/s) — caps how fast the arm can move per step
MAX_JOINT_VEL_RAD = 0.4    # rad/s (~23 deg/s); lowered from 0.8 for smoother motion


# Master-arm gripper channel (index 6): servo[6] is the angle OFFSET from home
# (zero-referenced, same convention as joints 0-5).
# Move servo past GRIPPER_OPEN_DEG → open;  past GRIPPER_CLOSE_DEG → close.
# Dead zone in between keeps current state.  Tune these to match your master arm travel.
GRIPPER_OPEN_DEG  =  -7.0   # deg offset from home → open  (tune down if hard to trigger)
GRIPPER_CLOSE_DEG = -17.0  # deg offset from home → close (large negative = intentional only)

CONTROL_HZ = 60  # main joint control loop rate
GRIPPER_HZ = 10  # gripper command rate

# ══════════════════════════════ Weiss CRG 30-050 ══════════════════════════════


class WeissCRGGripper:
    """
    USB serial driver for Weiss IEG76 / CRG gripper (DC-IOLink adapter).

    Protocol ref: https://github.com/ipa320/weiss_gripper_ieg76
    @PDIN=[B0,B1,B2,B3]: pos_mm = ((B0<<8)|B1)/100,  B3 = status flags
        bit1 OPEN  |  bit2 CLOSED  |  bit3 HOLDING  |  bit4 FAULT
    Key commands:
        PDOUT=[07,00]  reference/home (closes then opens)
        PDOUT=[02,00]  open
        PDOUT=[03,00]  close
        PDOUT=[00,00]  deactivate
    Mandatory init sequence before any PDOUT:
        ID? → FALLBACK(1) → MODE? → RESTART() → OPERATE() → PDOUT=[00,00]
    """

    FLAG_OPEN    = 1
    FLAG_CLOSED  = 2
    FLAG_HOLDING = 3
    FLAG_FAULT   = 4

    def __init__(self, port: str = GRIPPER_PORT, baudrate: int = GRIPPER_BAUDRATE, logger=None):
        self._lock = threading.Lock()
        self._logger = logger
        self._ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.2)
        self._position_mm = 0.0
        self._flags = 0
        self._log_info(f"Opened {port} @ {baudrate} baud")
        self._initialise()

    def _log_info(self, msg: str):
        if self._logger:
            self._logger.info(f"[WeissCRG] {msg}")

    def _log_warn(self, msg: str):
        if self._logger:
            self._logger.warning(f"[WeissCRG] {msg}")

    def _parse_pdin(self, line: str) -> bool:
        try:
            inner = line[7:].split("]")[0]
            parts = [int(x, 16) for x in inner.split(",")]
            self._position_mm = ((parts[0] << 8) | parts[1]) / 100.0
            self._flags = parts[3] if len(parts) >= 4 else 0
            return True
        except Exception:
            return False

    def _send(self, cmd: str, wait: float = 0.3) -> None:
        with self._lock:
            try:
                self._ser.reset_input_buffer()
                self._ser.write((cmd + "\n").encode("ascii"))
            except Exception as e:
                self._log_warn(f"Serial error on '{cmd}': {e}")
        time.sleep(wait)

    def _read_pdin(self, timeout: float = 1.0) -> bool:
        t0 = time.monotonic()
        with self._lock:
            saved, self._ser.timeout = self._ser.timeout, 0.15
        try:
            while time.monotonic() - t0 < timeout:
                with self._lock:
                    line = self._ser.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("@PDIN=[") and self._parse_pdin(line):
                    return True
        finally:
            with self._lock:
                self._ser.timeout = saved
        return False

    def _wait_flag(self, flag_bit: int, timeout: float = 6.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self._read_pdin(0.5) and (self._flags & (1 << flag_bit)):
                return True
        return False

    def _initialise(self) -> None:
        """Mandatory startup sequence before any PDOUT command."""
        for cmd in ["ID?", "ID?", "FALLBACK(1)", "MODE?", "RESTART()", "OPERATE()"]:
            self._send(cmd, 0.5)
        self._send("PDOUT=[00,00]", 0.5)
        self._log_info("Initialisation complete.")

    def get_width(self) -> float:
        self._read_pdin(timeout=0.5)
        return self._position_mm

    def home(self) -> bool:
        """Reference cycle: closes then opens fully."""
        self._log_info("Homing (reference cycle)...")
        self._send("PDOUT=[07,00]", 0.2)
        self._wait_flag(self.FLAG_OPEN, timeout=10.0)
        self._log_info("Homing done, gripper open.")
        return True

    def move_to_pos(self, width_mm: float) -> bool:
        """Open (width_mm > threshold) or close the gripper."""
        if width_mm > GRIPPER_OPEN_THRESHOLD_MM:
            self._send("PDOUT=[02,00]", 0.2)   # open
        else:
            self._send("PDOUT=[03,00]", 0.2)   # close
        return True

    def close(self):
        self._send("PDOUT=[00,00]", 0.3)
        self._send("FALLBACK(1)", 0.3)
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
        self.get_logger().info(f"[UR5Teleop] (1/4) Connecting RTDE control to {UR5_IP}...")
        self.rtde_c = rtde_control.RTDEControlInterface(UR5_IP)
        self.get_logger().info(f"[UR5Teleop] (2/4) Connecting RTDE receive to {UR5_IP}...")
        self.rtde_r = rtde_receive.RTDEReceiveInterface(UR5_IP)
        self.get_logger().info("[UR5Teleop] RTDE connected.")

        # ── Weiss CRG 30-050 ──────────────────────────────────────
        self.get_logger().info(f"[UR5Teleop] (3/4) Initialising gripper on {GRIPPER_PORT} (takes ~5 s)...")
        self.gripper = WeissCRGGripper(GRIPPER_PORT, GRIPPER_BAUDRATE, logger=self.get_logger())
        self.get_logger().info("[UR5Teleop] Gripper init done. Running home cycle (takes ~10 s)...")
        self.gripper.home()
        self.get_logger().info("[UR5Teleop] Gripper homed.")

        # ── UR5 home position ─────────────────────────────────────
        self.home_rad = np.radians(UR5_HOME_DEG)
        self.get_logger().info(f"[UR5Teleop] (4/4) Moving UR5 to home: {UR5_HOME_DEG} deg (slow moveJ)...")
        self.rtde_c.moveJ(self.home_rad.tolist(), speed=0.3, acceleration=0.3)
        self.get_logger().info("[UR5Teleop] UR5 at home position.")

        # ── shared command state ──────────────────────────────────
        self._lock = threading.Lock()
        self._cmd_joints_rad = self.home_rad.copy()
        self._cmd_gripper_mm = GRIPPER_MAX_MM  # gripper starts open after home()
        self._last_cmd_rad = self.home_rad.copy()
        self._servo_warned = False

        # ── ROS 2 interfaces ──────────────────────────────────────
        self._pub = self.create_publisher(Float64MultiArray, "/robot_action", 10)
        self.create_subscription(Float64MultiArray, "/servo_angles", self._servo_callback, 10)

        # ── gripper command thread ────────────────────────────────
        self._gripper_thread = threading.Thread(target=self._gripper_loop, daemon=True)
        self._gripper_thread.start()

        self.get_logger().info(
            f"[UR5Teleop] *** READY *** — streaming servoJ at {CONTROL_HZ} Hz, "
            f"gripper at {GRIPPER_HZ} Hz.  Move the master arm now."
        )

    # ── angle mapping ─────────────────────────────────────────────────────────

    def _map_joints(self, servo_deg: np.ndarray) -> np.ndarray:
        reordered = servo_deg[JOINT_MAP]
        delta_rad = np.deg2rad(reordered) * np.array(JOINT_SCALE)
        return self.home_rad + delta_rad

    def _map_gripper(self, servo_deg: float) -> float | None:
        """Absolute-offset gripper: servo[6] is zero-referenced (offset from home).
        Returns GRIPPER_MAX_MM (open), 0.0 (close), or None (dead zone → keep state)."""
        if servo_deg > GRIPPER_OPEN_DEG:
            return GRIPPER_MAX_MM
        elif servo_deg < GRIPPER_CLOSE_DEG:
            return 0.0
        return None  # dead zone — keep current state

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
            if gripper_mm is not None:   # None = dead zone, keep current
                self._cmd_gripper_mm = gripper_mm

    # ── gripper thread ────────────────────────────────────────────────────────

    def _gripper_loop(self):
        dt = 1.0 / GRIPPER_HZ
        # Explicitly open on startup.
        self.gripper.move_to_pos(GRIPPER_MAX_MM)
        last_open = True
        while rclpy.ok():
            with self._lock:
                target_mm = self._cmd_gripper_mm
            want_open = target_mm > GRIPPER_OPEN_THRESHOLD_MM
            if want_open != last_open:
                self.gripper.move_to_pos(target_mm)
                last_open = want_open
                self.get_logger().info(f"[UR5Teleop] Gripper {'opening' if want_open else 'closing'}")
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
                    0,
                    0,
                    SERVO_J_TIME,
                    SERVO_J_LOOKAHEAD,
                    SERVO_J_GAIN,
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
