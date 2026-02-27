#!/usr/bin/env python3
import os
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

# scservo_sdk/ lives in the same directory as this script.
# Python automatically prepends the script's directory to sys.path when
# the script is run with an absolute path (e.g. from run_ur5_nodes.sh),
# so no manual sys.path manipulation is needed.  The explicit insert below
# is a safeguard for the rare case where the interpreter doesn't do so.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from scservo_sdk import *  # noqa: E402


class ServoReaderNode(Node):
    def __init__(self):
        super().__init__("servo_reader_node")

        self.pub = self.create_publisher(Float64MultiArray, "/servo_angles", 10)

        self.portHandler = PortHandler("/dev/ttyUSB0")
        self.packetHandler = sms_sts(self.portHandler)
        self.portHandler.openPort()
        self.portHandler.setBaudRate(1000000)

        self.get_logger().info("Serial port opened")

        self.groupSyncRead = GroupSyncRead(self.packetHandler, SMS_STS_PRESENT_POSITION_L, 4)
        self.gripper_range = 0.48
        self.zero_angles = [0.0] * 7
        self._init_servos()

    def _init_servos(self):
        self.get_logger().info("Calibrating servo half positions... DON'T MOVE!!!")
        for i in range(7):
            scs_id = i + 1
            self.packetHandler.unLockEprom(scs_id)
            time.sleep(0.1)

            self.packetHandler.write2ByteTxRx(scs_id, SMS_STS_OFS_L, 0)
            time.sleep(0.1)

            raw_pos, result, error = self.packetHandler.read2ByteTxRx(scs_id, SMS_STS_PRESENT_POSITION_L)

            homing_offset = raw_pos - 2047
            encoded_offset = 1 << 11 | abs(homing_offset) if homing_offset < 0 else homing_offset

            comm, error = self.packetHandler.write2ByteTxRx(scs_id, SMS_STS_OFS_L, encoded_offset)
            if error == 0:
                self.get_logger().info(f"Succeeded to set the half position for id:{scs_id}")
            time.sleep(0.1)

            self.packetHandler.LockEprom(scs_id)
            time.sleep(0.1)

        for i in range(7):
            scs_id = i + 1
            scs_addparam_result = self.groupSyncRead.addParam(scs_id)
            if not scs_addparam_result:
                self.get_logger().warning(f"[ID:{scs_id:03d}] groupSyncRead addparam failed")

        scs_comm_result = self.groupSyncRead.txRxPacket()
        if scs_comm_result != COMM_SUCCESS:
            self.get_logger().warning(f"{self.packetHandler.getTxRxResult(scs_comm_result)}")

        for i in range(7):
            scs_id = i + 1
            scs_data_result, scs_error = self.groupSyncRead.isAvailable(scs_id, SMS_STS_PRESENT_POSITION_L, 4)
            if scs_data_result:
                scs_present_position = self.groupSyncRead.getData(scs_id, SMS_STS_PRESENT_POSITION_L, 2)
                self.zero_angles[i] = scs_present_position
            else:
                self.zero_angles[i] = 2047
                self.get_logger().warning(f"[ID:{scs_id:03d}] groupSyncRead getdata failed")
                continue
            if scs_error != 0:
                self.get_logger().warning(f"{self.packetHandler.getRxPacketError(scs_error)}")

        self.get_logger().info(f"Zero positions (raw): {self.zero_angles}")
        self.groupSyncRead.clearParam()

    def run(self):
        dt = 1.0 / 50  # 50 Hz
        num_interp = 5
        step_size = 1
        angle_offset = [0.0] * 7
        target_angle_offset = [0.0] * 7

        while rclpy.ok():
            for i in range(7):
                scs_addparam_result = self.groupSyncRead.addParam(i + 1)
                if not scs_addparam_result:
                    self.get_logger().warning(f"[ID:{i + 1:03d}] groupSyncRead addparam failed")

            scs_comm_result = self.groupSyncRead.txRxPacket()
            if scs_comm_result != COMM_SUCCESS:
                self.get_logger().warning("GroupSyncRead failed")

            for i in range(7):
                scs_id = i + 1
                scs_data_result, scs_error = self.groupSyncRead.isAvailable(scs_id, SMS_STS_PRESENT_POSITION_L, 4)
                if scs_data_result:
                    current_pos = self.groupSyncRead.getData(scs_id, SMS_STS_PRESENT_POSITION_L, 2)
                    if current_pos is not None:
                        new_angle = (current_pos - self.zero_angles[i]) / 4096.0 * 360.0
                        if abs(new_angle - target_angle_offset[i]) > step_size:
                            target_angle_offset[i] = new_angle
                    else:
                        self.get_logger().warning(f"Can't read servo {scs_id}")
                else:
                    self.get_logger().warning(f"Failed to get data for ID {scs_id}")

            self.groupSyncRead.clearParam()

            # Interpolate to approach target angle
            for _step in range(num_interp):
                for i in range(7):
                    delta = target_angle_offset[i] - angle_offset[i]
                    angle_offset[i] += delta * 0.2
                msg = Float64MultiArray()
                msg.data = list(angle_offset)
                self.pub.publish(msg)
                time.sleep(dt / num_interp)


def main():
    rclpy.init()
    node = ServoReaderNode()

    # Spin in background thread so ROS 2 callbacks are processed;
    # run() holds the publish loop in the main thread.
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
