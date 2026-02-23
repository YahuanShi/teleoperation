#!/usr/bin/env python3
"""
UR5 + Weiss Robotics CRG 30-050 State Publisher

Reads the actual joint configuration and gripper width from the robot and
publishes them on /robot_state for the episode recorder.

Published topic:
    /robot_state  (Float64MultiArray, 7 floats:
                   joints 0-5 in degrees (actual) + gripper width 0-1 normalised)

Hardware:
    - UR5 connected via Ethernet (RTDE)
    - Weiss Robotics CRG 30-050 connected via USB cable (/dev/ttyACM0)

Dependencies:
    pip install ur-rtde pyserial
"""

import rospy
from std_msgs.msg import Float64MultiArray
import numpy as np
import threading
import struct

try:
    import rtde_receive
except ImportError:
    raise ImportError("ur_rtde not installed. Run: pip install ur-rtde")

try:
    import serial
except ImportError:
    raise ImportError("pyserial not installed. Run: pip install pyserial")

# ══════════════════════════════ Configuration ══════════════════════════════
# Keep in sync with servo2ur5.py

UR5_IP           = "192.168.1.100"   # ← change to your UR5 controller IP
GRIPPER_PORT     = "/dev/ttyACM0"    # ← USB serial port for CRG 30-050
GRIPPER_BAUDRATE = 115200
GRIPPER_MAX_MM   = 30.0              # CRG 30-050 max opening (must match servo2ur5.py)

PUBLISH_HZ = 10   # /robot_state publish rate

# ══════════════════════════════ Weiss CRG 30-050 (read-only) ══════════════════

class WeissCRGReader:
    """
    Minimal USB serial reader for the Weiss Robotics CRG 30-050.
    Queries only the current opening width (CMD_GET_WIDTH).
    """

    CMD_GET_WIDTH = 0x43
    E_SUCCESS     = 0x00

    def __init__(self, port: str = GRIPPER_PORT, baudrate: int = GRIPPER_BAUDRATE):
        self._lock = threading.Lock()
        self._ser  = serial.Serial(port=port, baudrate=baudrate, timeout=1.0)
        rospy.loginfo(f"[UR5Pub] Gripper reader opened {port} @ {baudrate} baud")

    @staticmethod
    def _crc16(data: bytes) -> int:
        crc = 0xFFFF
        for b in data:
            crc ^= b
            for _ in range(8):
                crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
        return crc & 0xFFFF

    def _build_packet(self, cmd: int, payload: bytes = b'') -> bytes:
        header = struct.pack('<BBHH', 0xAA, 0xAA, cmd, len(payload))
        body   = header + payload
        return body + struct.pack('<H', self._crc16(body))

    def _recv_all(self, n: int) -> bytes:
        buf = b''
        while len(buf) < n:
            chunk = self._ser.read(n - len(buf))
            if not chunk:
                raise TimeoutError(f"[UR5Pub] Serial read timeout ({len(buf)}/{n} bytes)")
            buf += chunk
        return buf

    def get_width(self) -> float:
        """Return current opening width in mm, or -1.0 on error."""
        with self._lock:
            try:
                self._ser.reset_input_buffer()
                self._ser.write(self._build_packet(self.CMD_GET_WIDTH))
                hdr  = self._recv_all(6)
                _, _, _, psize = struct.unpack('<BBHH', hdr)
                rest    = self._recv_all(psize + 2)
                payload = rest[:psize]
                status  = payload[0] if payload else 0xFF
                if status == self.E_SUCCESS and len(payload) >= 5:
                    return float(struct.unpack('<f', payload[1:5])[0])
            except Exception as e:
                rospy.logwarn_throttle(5.0, f"[UR5Pub] Gripper read error: {e}")
        return -1.0

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

# ══════════════════════════════ State Publisher Node ══════════════════════════

class UR5StatePublisher:
    """
    Publishes the actual UR5 joint angles and CRG 30-050 gripper width
    on /robot_state for the episode recorder.
    """

    def __init__(self):
        rospy.init_node("ur5_state_pub")

        # ── RTDE receive interface ────────────────────────────────
        rospy.loginfo(f"[UR5Pub] Connecting RTDE receive to {UR5_IP}...")
        self.rtde_r = rtde_receive.RTDEReceiveInterface(UR5_IP)
        rospy.loginfo("[UR5Pub] RTDE receive connected.")

        # ── Weiss CRG reader ──────────────────────────────────────
        rospy.loginfo(f"[UR5Pub] Opening gripper reader on {GRIPPER_PORT}...")
        self.gripper_reader = WeissCRGReader(GRIPPER_PORT, GRIPPER_BAUDRATE)

        # ── ROS publisher ─────────────────────────────────────────
        self._pub = rospy.Publisher('/robot_state', Float64MultiArray, queue_size=10)
        rospy.Timer(rospy.Duration(1.0 / PUBLISH_HZ), self._publish_state)

        rospy.loginfo(f"[UR5Pub] Publishing /robot_state at {PUBLISH_HZ} Hz.")

    def _publish_state(self, _event):
        try:
            # Actual joint angles (radians) → degrees
            q_rad = self.rtde_r.getActualQ()
            q_deg = np.degrees(q_rad).tolist()

            # Gripper actual width, normalised to [0, 1]
            width_mm  = self.gripper_reader.get_width()
            grip_norm = float(np.clip(width_mm / GRIPPER_MAX_MM, 0.0, 1.0)) \
                        if width_mm >= 0.0 else 0.0

            state = q_deg + [grip_norm]
            self._pub.publish(Float64MultiArray(data=state))
            rospy.logdebug(f"[UR5Pub] state: {[f'{v:.2f}' for v in state]}")

        except Exception as e:
            rospy.logwarn_throttle(2.0, f"[UR5Pub] State publish error: {e}")


# ══════════════════════════════ Entry point ══════════════════════════════

if __name__ == "__main__":
    try:
        node = UR5StatePublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
