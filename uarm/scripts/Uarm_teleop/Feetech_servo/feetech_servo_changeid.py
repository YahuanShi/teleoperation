# set new id for servo
import sys

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

# id for setting. Please set id from #1 to #7 (or #8 for Config 3)
new_id = 8
print(f"Set the servo's id to: {new_id} (Make sure only one servo is connected)...")

# Unlock EPROM
packet_handler.unLockEprom(BROADCAST_ID)
time.sleep(0.1)

# Write new id in
result, error = packet_handler.write1ByteTxRx(BROADCAST_ID, SMS_STS_ID, new_id)
if result != COMM_SUCCESS:
    print(f"Failed to set id: {packet_handler.getTxRxResult(result)}")
    # Lock EPROM
    packet_handler.LockEprom(BROADCAST_ID)
    port_handler.closePort()
    sys.exit()

print(f"Succeed to set id: {new_id}")
time.sleep(0.1)

# Lock EPROM
packet_handler.LockEprom(new_id)
time.sleep(0.1)
