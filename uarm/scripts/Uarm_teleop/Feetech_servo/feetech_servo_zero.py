# group read servo position

import sys

import numpy as np

sys.path.append("..")
from scservo_sdk import *

port_handler = PortHandler("/dev/ttyUSB0")
packet_handler = sms_sts(port_handler)

# Open port
if port_handler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    sys.exit()


# Set port baudrate 1000000
if port_handler.setBaudRate(1000000):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    sys.exit()

# Set all servos' half position
for scs_id in range(1, 9):
    # Unlock EPROM
    packet_handler.unLockEprom(scs_id)
    time.sleep(0.1)

    # First write 0 in
    comm, error = packet_handler.write2ByteTxRx(scs_id, SMS_STS_OFS_L, 0)
    time.sleep(0.1)

    # Read present position
    raw_pos, result, error = packet_handler.read2ByteTxRx(scs_id, SMS_STS_PRESENT_POSITION_L)
    if result != COMM_SUCCESS:
        print(f"Failed to read the position: {packet_handler.getTxRxResult(result)}")

    homing_offset = raw_pos - 2047
    encoded_offset = (1 << 11) | abs(homing_offset) if homing_offset < 0 else homing_offset  # highest bit = sign

    # Write the offset in
    comm, error = packet_handler.write2ByteTxRx(scs_id, SMS_STS_OFS_L, encoded_offset)
    if error == 0:
        print(f"Succeeded to set the half position for id:{scs_id:d}")
    time.sleep(0.1)

    # Lock EPROM
    packet_handler.LockEprom(scs_id)
    time.sleep(0.1)


# Group read
group_sync_read = GroupSyncRead(packet_handler, SMS_STS_PRESENT_POSITION_L, 4)
angle_pos = np.zeros(8)


while 1:
    for scs_id in range(1, 9):
        # Add parameter storage for SCServo#1~10 present position value
        scs_addparam_result = group_sync_read.addParam(scs_id)
        if not scs_addparam_result:
            print(f"[ID:{scs_id:03d}] groupSyncRead addparam failed")

    scs_comm_result = group_sync_read.txRxPacket()
    if scs_comm_result != COMM_SUCCESS:
        print(f"{packet_handler.getTxRxResult(scs_comm_result)}")

    for scs_id in range(1, 9):
        # Check if groupsyncread data of SCServo#1~10 is available
        scs_data_result, scs_error = group_sync_read.isAvailable(scs_id, SMS_STS_PRESENT_POSITION_L, 4)
        if scs_data_result:
            # Get SCServo#scs_id present position value
            scs_present_position = group_sync_read.getData(scs_id, SMS_STS_PRESENT_POSITION_L, 2)
            angle_pos[scs_id - 1] = scs_present_position
        else:
            print(f"[ID:{scs_id:03d}] groupSyncRead getdata failed")
            continue
        if scs_error != 0:
            print(f"{packet_handler.getRxPacketError(scs_error)}")
    print(angle_pos)
    group_sync_read.clearParam()
    # time.sleep(0.1)
