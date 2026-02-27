#!/usr/bin/env python3
"""
UR5 + Weiss Robotics CRG 30-050 Teleoperation Node  (ROS 2 Humble)

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

import struct
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

# CRG 30-050: 30 mm total stroke, 50 N max force
GRIPPER_MAX_MM = 30.0
GRIPPER_SPEED_MM_S = 50.0
GRIPPER_FORCE_N = 40.0

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
    USB serial driver for the Weiss Robotics CRG 30-050 gripper.

    Uses the Weiss WSG binary command protocol over a virtual serial port
    (USB CDC-ACM).  Packet layout:

        preamble  : 2 bytes   0xAA 0xAA
        command   : 2 bytes   uint16 little-endian
        size      : 2 bytes   uint16 little-endian  (payload byte count)
        payload   : N bytes
        checksum  : 2 bytes   CRC-16/IBM little-endian

    Response payload byte 0 is always a status code (0x00 = success).
    """

    CMD_HOMING = 0x20
    CMD_MOVE_TO_POS = 0x21
    CMD_GET_STATE = 0x40
    CMD_GET_WIDTH = 0x43
    E_SUCCESS = 0x00

    def __init__(self, port: str = GRIPPER_PORT, baudrate: int = GRIPPER_BAUDRATE, logger=None):
        self._lock = threading.Lock()
        self._logger = logger
        self._ser = serial.Serial(port=port, baudrate=baudrate, timeout=1.0)
        self._log_info(f"Opened {port} @ {baudrate} baud")

    def _log_info(self, msg: str):
        if self._logger:
            self._logger.info(f"[WeissCRG] {msg}")

    def _log_warn(self, msg: str):
        if self._logger:
            self._logger.warning(f"[WeissCRG] {msg}")

    @staticmethod
    def _crc16(data: bytes) -> int:
        crc = 0xFFFF
        for b in data:
            crc ^= b
            for _ in range(8):
                crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
        return crc & 0xFFFF

    def _build_packet(self, cmd: int, payload: bytes = b"") -> bytes:
        header = struct.pack("<BBHH", 0xAA, 0xAA, cmd, len(payload))
        body = header + payload
        return body + struct.pack("<H", self._crc16(body))

    def _recv_all(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self._ser.read(n - len(buf))
            if not chunk:
                raise TimeoutError(f"Serial read timeout (expected {n} bytes, got {len(buf)})")
            buf += chunk
        return buf

    def _recv_packet(self) -> tuple:
        hdr = self._recv_all(6)
        _, _, cmd_id, psize = struct.unpack("<BBHH", hdr)
        rest = self._recv_all(psize + 2)
        payload = rest[:psize]
        status = payload[0] if payload else self.E_SUCCESS
        return cmd_id, status, payload[1:]

    def _send_cmd(self, cmd: int, payload: bytes = b"") -> tuple:
        with self._lock:
            try:
                self._ser.reset_input_buffer()
                self._ser.write(self._build_packet(cmd, payload))
                _, status, resp = self._recv_packet()
                return status, resp
            except Exception as e:
                self._log_warn(f"Serial error: {e}")
                return 0xFF, b""

    def home(self) -> bool:
        payload = struct.pack("<B", 0)
        status, _ = self._send_cmd(self.CMD_HOMING, payload)
        ok = status == self.E_SUCCESS
        if ok:
            self._log_info("Homing complete")
        else:
            self._log_warn(f"Homing returned status 0x{status:02X}")
        return ok

    def move_to_pos(
        self, width_mm: float, speed_mm_s: float = GRIPPER_SPEED_MM_S, force_n: float = GRIPPER_FORCE_N
    ) -> bool:
        width_mm = float(np.clip(width_mm, 0.0, GRIPPER_MAX_MM))
        payload = struct.pack("<fff", width_mm, speed_mm_s, force_n)
        status, _ = self._send_cmd(self.CMD_MOVE_TO_POS, payload)
        return status == self.E_SUCCESS

    def get_width(self) -> float:
        status, data = self._send_cmd(self.CMD_GET_WIDTH)
        if status == self.E_SUCCESS and len(data) >= 4:
            return float(struct.unpack("<f", data[:4])[0])
        return -1.0

    def close(self):
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
        while rclpy.ok():
            with self._lock:
                target_mm = self._cmd_gripper_mm
            self.gripper.move_to_pos(target_mm)
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
