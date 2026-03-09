import serial
import struct
import time

def crc16(data):
    """Weiss Gripper 标准 CRC16 校验 (Modbus 风格)"""
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
    """构建并发送数据包"""
    # Header: 0xAA 0xAA 0xAA | CMD | Payload_Len (Little Endian)
    header = struct.pack("<BBBBH", 0xAA, 0xAA, 0xAA, cmd_hex, len(payload))
    packet = header + payload
    packet += struct.pack("<H", crc16(packet))
    
    print(f"-> 发送指令 [0x{cmd_hex:02X}]: {packet.hex(' ')}")
    ser.reset_input_buffer()
    ser.write(packet)
    ser.flush() # 确保数据完全写出

def test_gripper(port="/dev/ttyACM0"):
    print(f"=== 开始夹爪通信测试: {port} ===")
    try:
        # 关键点：开启 DTR 和 RTS，这对于 STM32 USB 虚拟串口至关重要
        ser = serial.Serial(
            port=port, 
            baudrate=115200, 
            timeout=2.0, 
            rtscts=False, 
            dsrdtr=False
        )
        ser.dtr = True  # 激活数据终端就绪
        ser.rts = True  # 激活请求发送
        time.sleep(0.5) # 给固件一点响应时间

        # 1. 尝试获取状态 (不涉及物理移动，安全测试)
        print("\n[步骤 1] 测试 GET_STATE (0x40)")
        send_cmd(ser, 0x40)
        response = ser.read(64)
        
        if len(response) > 0:
            print(f"<- 收到反馈 ({len(response)} bytes): {response.hex(' ')}")
            print("✅ 通信正常！硬件层已打通。")
        else:
            print("❌ 错误：未收到任何反馈 (Got 0 bytes)。")
            print("提示：请检查 24V 供电！流光灯闪烁通常意味着逻辑层在跑，但功率层没准备好。")
            return

        # 2. 尝试 Homing (会有物理动作)
        user_input = input("\n[步骤 2] 是否尝试 Homing 回零 (夹爪会移动)? (y/n): ")
        if user_input.lower() == 'y':
            # Homing 指令 0x20，payload 为 0x00
            send_cmd(ser, 0x20, struct.pack("<B", 0))
            print("正在等待回零完成 (最多 15 秒)...")
            
            # 临时增加超时时间给物理动作
            ser.timeout = 15.0
            resp_homing = ser.read(64)
            if resp_homing:
                print(f"<- Homing 反馈: {resp_homing.hex(' ')}")
                print("✅ Homing 动作执行成功！")
            else:
                print("❌ Homing 超时或无反馈。")

        ser.close()
    except Exception as e:
        print(f"❌ 运行异常: {e}")

if __name__ == "__main__":
    test_gripper()