#!/usr/bin/env python3
import re
import time

import rclpy
from rclpy.node import Node
import serial
from std_msgs.msg import Float64MultiArray

# Deadband: ignore movements smaller than this (degrees) to suppress sensor noise.
DEADBAND_DEG = 0.8

# Exponential smoothing factor for angle output (0 < alpha <= 1).
# Higher = faster response, lower = smoother.  0.5 is a good starting point.
SMOOTH_ALPHA = 0.5

# Jump filter: discard readings that jump more than this from the last known value.
JUMP_LIMIT_DEG = 60.0

# Number of joints to read (excluding gripper index 6 which is read separately).
N_JOINTS = 6
GRIPPER_IDX = 6


class ServoReaderNode(Node):
    def __init__(self):
        super().__init__("servo_reader_node")

        self.pub = self.create_publisher(Float64MultiArray, "/servo_angles", 10)

        serial_port = self.declare_parameter("serial_port", "/dev/ttyUSB0").value
        baudrate    = self.declare_parameter("baudrate", 115200).value

        self.ser = serial.Serial(serial_port, baudrate, timeout=0.1)
        self.get_logger().info(f"Serial port {serial_port} opened @ {baudrate} baud")

        self.zero_angles   = [0.0] * 7   # calibrated zero per servo
        self.angle_smooth  = [0.0] * 7   # smoothed output (published)
        self.angle_raw     = [0.0] * 7   # last valid raw reading

        self._init_servos()

        # Publish at ~20 Hz (all 6 joints + gripper read per tick = 7 × 8 ms ≈ 56 ms/cycle)
        self.create_timer(1.0 / 20.0, self._timer_cb)

    # ── Serial helpers ─────────────────────────────────────────────────────

    def send_command(self, cmd: str) -> str:
        self.ser.write(cmd.encode("ascii"))
        time.sleep(0.008)
        return self.ser.read_all().decode("ascii", errors="ignore")

    def pwm_to_angle(self, response_str: str,
                     pwm_min=500, pwm_max=2500, angle_range=270) -> float | None:
        match = re.search(r"P(\d{4})", response_str)
        if not match:
            return None
        pwm_val = int(match.group(1))
        return (pwm_val - pwm_min) / (pwm_max - pwm_min) * angle_range

    # ── Initialisation ─────────────────────────────────────────────────────

    def _init_servos(self):
        self.send_command("#000PVER!")
        for i in range(7):
            self.send_command("#000PCSK!")
            self.send_command(f"#{i:03d}PULK!")
            response = self.send_command(f"#{i:03d}PRAD!")
            angle = self.pwm_to_angle(response.strip())
            self.zero_angles[i]  = angle if angle is not None else 0.0
            self.angle_raw[i]    = 0.0
            self.angle_smooth[i] = 0.0
        self.get_logger().info("Servo initial angle calibration completed")

    # ── 20 Hz timer — read ALL joints every tick ────────────────────────────

    def _timer_cb(self):
        for i in range(7):
            response = self.send_command(f"#{i:03d}PRAD!")
            angle = self.pwm_to_angle(response.strip())
            if angle is None:
                self.get_logger().warn(
                    f"Servo {i} no response", throttle_duration_sec=2.0)
                continue

            delta_from_zero = angle - self.zero_angles[i]

            # Jump filter
            if abs(delta_from_zero - self.angle_raw[i]) > JUMP_LIMIT_DEG:
                self.get_logger().warn(
                    f"Servo {i} jump suppressed: "
                    f"{self.angle_raw[i]:.1f} -> {delta_from_zero:.1f} deg",
                    throttle_duration_sec=1.0)
                continue

            # Deadband: only update target if movement exceeds threshold
            if abs(delta_from_zero - self.angle_raw[i]) >= DEADBAND_DEG:
                self.angle_raw[i] = delta_from_zero

            # Exponential smoothing toward raw target
            self.angle_smooth[i] += SMOOTH_ALPHA * (
                self.angle_raw[i] - self.angle_smooth[i]
            )

        self.pub.publish(Float64MultiArray(data=list(self.angle_smooth)))


def main(args=None):
    rclpy.init(args=args)
    node = ServoReaderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
