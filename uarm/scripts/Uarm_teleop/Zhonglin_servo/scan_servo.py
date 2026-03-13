import serial
import time

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

def scan():
    # Scan ID 0 to 253 by sending PVER command and checking for responses
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05) as ser:
        print("正在扫描 ID 0 到 253... 请稍候")
        for i in range(254):
            cmd = f"#{i:03d}PVER!\r\n".encode('ascii')
            ser.write(cmd)
            time.sleep(0.02)
            
            if ser.in_waiting:
                response = ser.read_all().decode(errors="ignore").strip()
                print(f"--- [ 发现目标 ] ---")
                print(f"ID: {i}")
                print(f"回复内容: {response}")
                print(f"------------------")
        print("扫描结束。")

if __name__ == "__main__":
    scan()