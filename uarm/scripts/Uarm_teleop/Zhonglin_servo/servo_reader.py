#!/usr/bin/env python3
import re
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import serial


class ServoReaderNode(Node):
    def __init__(self):
        super().__init__("servo_reader_node")

        self.pub = self.create_publisher(Float64MultiArray, "/servo_angles", 10)

        serial_port = self.declare_parameter("serial_port", "/dev/ttyUSB0").value
        baudrate    = self.declare_parameter("baudrate", 115200).value

        self.ser = serial.Serial(serial_port, baudrate, timeout=0.1)
        self.get_logger().info(f"Serial port {serial_port} opened @ {baudrate} baud")

        self.gripper_range = 0.48
        self.zero_angles = [0.0] * 7
        self._init_servos()

        # Run the read/publish loop in a ROS 2 timer at 50 Hz
        self.angle_offset        = [0.0] * 7
        self.target_angle_offset = [0.0] * 7
        self._interp_step = 0
        self._num_interp  = 5
        self._step_size   = 1

        self._servo_index = 0   # rolling servo read index
        self.create_timer(1.0 / 50.0, self._timer_cb)

    # ── Serial helpers ──────────────────────────────────────────────────────

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

    # ── Initialisation ──────────────────────────────────────────────────────

    def _init_servos(self):
        self.send_command("#000PVER!")
        for i in range(7):
            self.send_command("#000PCSK!")
            self.send_command(f"#{i:03d}PULK!")
            response = self.send_command(f"#{i:03d}PRAD!")
            angle = self.pwm_to_angle(response.strip())
            self.zero_angles[i] = angle if angle is not None else 0.0
        self.get_logger().info("Servo initial angle calibration completed")

    # ── 50 Hz timer ────────────────────────────────────────────────────────

    def _timer_cb(self):
        """
        Interleave reading and publishing so the 8 ms serial wait per servo
        doesn't block the ROS executor.  One servo is read per tick; after
        all 7 are read the interpolation step fires.
        """
        i = self._servo_index
        response = self.send_command(f"#{i:03d}PRAD!")
        angle = self.pwm_to_angle(response.strip())
        if angle is not None:
            new_angle = angle - self.zero_angles[i]
            if abs(new_angle - self.target_angle_offset[i]) > 90:
                self.get_logger().warn(
                    f"Servo {i} angle jump too large: "
                    f"{new_angle:.1f}° vs {self.target_angle_offset[i]:.1f}°"
                )
            elif abs(new_angle - self.target_angle_offset[i]) > self._step_size:
                self.target_angle_offset[i] = new_angle
        else:
            self.get_logger().warn(
                f"Servo {i} response error: {response.strip()}", throttle_duration_sec=1.0
            )

        self._servo_index = (self._servo_index + 1) % 7

        # Publish one interpolation step every tick
        for j in range(7):
            delta = self.target_angle_offset[j] - self.angle_offset[j]
            self.angle_offset[j] += delta * 0.2
        self.pub.publish(Float64MultiArray(data=list(self.angle_offset)))


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
