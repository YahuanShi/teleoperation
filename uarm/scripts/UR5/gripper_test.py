import serial
import struct
import time

def crc16(data):
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF

def send_cmd(ser, cmd_hex, payload=b""):
    header = struct.pack("<BBBBH", 0xAA, 0xAA, 0xAA, cmd_hex, len(payload))
    packet = header + payload
    packet += struct.pack("<H", crc16(packet))
    print(f"-> 发送 [0x{cmd_hex:02X}]: {packet.hex(' ')}")
    ser.reset_input_buffer()
    ser.write(packet)
    ser.flush()

try:
    ser = serial.Serial("/dev/ttyACM0", 115200, timeout=2.0)
    ser.dtr = True
    ser.rts = True
    time.sleep(0.5)

    print("--- 步骤 1: 检查状态 ---")
    send_cmd(ser, 0x40)
    resp = ser.read(64)
    if resp:
        print(f"<- 收到回复: {resp.hex(' ')}")
        
        print("\n--- 步骤 2: 尝试 Homing (回零) ---")
        send_cmd(ser, 0x20, struct.pack("<B", 0))
        ser.timeout = 15.0
        resp_h = ser.read(64)
        print(f"<- Homing 结果: {resp_h.hex(' ')}")
    else:
        print("❌ 未收到任何回复。请检查 24V 电源和 USB 线！")
    
    ser.close()
except Exception as e:
    print(f"❌ 运行失败: {e}")
