#!/usr/bin/env python3
"""
UR5 State Publisher

Published topics:
    /robot_state  (Float64MultiArray, 7 floats:
                   joints 0-5 in degrees (actual) + gripper width 0-1 normalised)
    /robot_vel    (Float64MultiArray, 7 floats:
                   joints 0-5 velocity in deg/s (actual, from RTDE getActualQd)
                   + index 6 = 0.0 placeholder (gripper has no velocity sensor))

Gripper state is derived from the /robot_action topic published by servo2ur5.py.
This avoids opening /dev/ttyACM0 a second time (servo2ur5.py already owns it).

Hardware:
    - UR5 connected via Ethernet (RTDE)

Dependencies:
    pip install ur-rtde
"""

import threading

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

try:
    import rtde_receive
except ImportError as err:
    raise ImportError("ur_rtde not installed. Run: pip install ur-rtde") from err

# ══════════════════════════════ Configuration ══════════════════════════════
# Keep UR5_IP in sync with servo2ur5.py

UR5_IP = "10.0.0.1"
PUBLISH_HZ = 10  # /robot_state publish rate

# ══════════════════════════════ State Publisher Node ══════════════════════════


class UR5StatePublisher(Node):
    """
    Publishes actual UR5 joint angles on /robot_state.
    Gripper normalised value (0=closed, 1=open) is read from /robot_action
    so that servo2ur5.py remains the sole owner of the gripper serial port.
    """

    def __init__(self):
        super().__init__("ur5_state_pub")

        # ── RTDE receive interface ────────────────────────────────
        self.get_logger().info(f"[UR5Pub] Connecting RTDE receive to {UR5_IP}...")
        self.rtde_r = rtde_receive.RTDEReceiveInterface(UR5_IP)
        self.get_logger().info("[UR5Pub] RTDE receive connected.")

        # ── Gripper state (received from /robot_action) ───────────
        self._gripper_norm = 0.0
        self._action_lock = threading.Lock()
        self.create_subscription(Float64MultiArray, "/robot_action", self._cb_action, 10)

        # ── ROS 2 publishers + timer ──────────────────────────────
        self._pub = self.create_publisher(Float64MultiArray, "/robot_state", 10)
        self._vel_pub = self.create_publisher(Float64MultiArray, "/robot_vel", 10)
        self.create_timer(1.0 / PUBLISH_HZ, self._publish_state)

        self.get_logger().info(f"[UR5Pub] Publishing /robot_state and /robot_vel at {PUBLISH_HZ} Hz.")

    def _cb_action(self, msg: Float64MultiArray):
        """Cache the commanded gripper value from servo2ur5.py."""
        if len(msg.data) >= 7:
            with self._action_lock:
                self._gripper_norm = float(msg.data[6])

    def _publish_state(self):
        try:
            q_rad = self.rtde_r.getActualQ()
            qd_rad = self.rtde_r.getActualQd()
            q_deg = np.degrees(q_rad).tolist()
            qd_deg = np.degrees(qd_rad).tolist()

            with self._action_lock:
                grip_norm = self._gripper_norm

            state_msg = Float64MultiArray()
            state_msg.data = [*q_deg, grip_norm]
            self._pub.publish(state_msg)

            vel_msg = Float64MultiArray()
            vel_msg.data = [*qd_deg, 0.0]  # index 6: gripper has no velocity sensor
            self._vel_pub.publish(vel_msg)

            self.get_logger().debug(f"[UR5Pub] state: {[f'{v:.2f}' for v in state_msg.data]}")

        except Exception as e:
            self.get_logger().warning(f"[UR5Pub] State publish error: {e}", throttle_duration_sec=2.0)


# ══════════════════════════════ Entry point ══════════════════════════════


def main():
    rclpy.init()
    node = UR5StatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
