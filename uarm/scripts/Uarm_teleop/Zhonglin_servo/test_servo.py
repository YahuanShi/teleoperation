#!/usr/bin/env python3
"""
test_servo.py — Zhonglin servo master arm hardware test (no ROS required).

Protocol: ASCII over serial (115200 baud)
    #IDXPRAD!  →  read current angle (response contains PXXXX, PWM 500–2500)
    #IDXPULK!  →  unlock servo
    #000PCSK!  →  check/ping
    PWM → angle: (pwm - 500) / 2000 * 270  degrees
    IDs: 000–006  (7 servos for 6 joints + gripper)

Steps:
    1. Scan IDs 0–6, report which servos respond
    2. Calibrate zero positions from current arm pose — keep arm STILL
    3. Print live joint angles at 50 Hz — move the arm to verify each joint

Usage:
    python3 test_servo.py [port]
    python3 test_servo.py /dev/ttyUSB0   # default

Press Ctrl+C to quit.
"""

import re
import sys
import time
import serial

PORT     = "/dev/ttyUSB0"
BAUDRATE = 115200
N_SERVOS = 7
SERVO_IDS = list(range(N_SERVOS))   # 0–6

PWM_MIN    = 500
PWM_MAX    = 2500
ANGLE_RANGE = 270.0   # degrees full span


def send(ser: serial.Serial, cmd: str) -> str:
    ser.write(cmd.encode("ascii"))
    time.sleep(0.008)
    return ser.read_all().decode("ascii", errors="ignore")


def pwm_to_angle(response: str) -> float | None:
    match = re.search(r"P(\d{4})", response)
    if not match:
        return None
    pwm = int(match.group(1))
    return (pwm - PWM_MIN) / (PWM_MAX - PWM_MIN) * ANGLE_RANGE


def read_angle(ser: serial.Serial, servo_id: int) -> float | None:
    resp = send(ser, f"#{servo_id:03d}PRAD!")
    return pwm_to_angle(resp.strip())


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else PORT
    print(f"Zhonglin servo master arm test  —  port: {port}")
    print("=" * 56)

    try:
        ser = serial.Serial(port=port, baudrate=BAUDRATE, timeout=0.1)
    except Exception as e:
        print(f"[FAIL] Cannot open {port}: {e}")
        sys.exit(1)
    print(f"[OK]  Port {port} opened @ {BAUDRATE} baud")

    time.sleep(0.3)
    send(ser, "#000PVER!")   # wake up / version check

    # ── Step 1: Scan servos ───────────────────────────────────────────────────
    print("\nScanning servo IDs 0–6...")
    found = []
    for sid in SERVO_IDS:
        send(ser, "#000PCSK!")
        send(ser, f"#{sid:03d}PULK!")
        angle = read_angle(ser, sid)
        if angle is not None:
            print(f"  ID {sid}: OK  ({angle:.1f}°)")
            found.append(sid)
        else:
            print(f"  ID {sid}: NO RESPONSE")

    if not found:
        print("\n[FAIL] No servos responded — check USB cable and power.")
        ser.close()
        sys.exit(1)

    if len(found) < N_SERVOS:
        missing = [sid for sid in SERVO_IDS if sid not in found]
        print(f"\n[WARN] Only {len(found)}/7 servos found. Missing IDs: {missing}")
    else:
        print(f"\n[OK]  All {N_SERVOS} servos found.")

    # ── Step 2: Calibrate zeros ───────────────────────────────────────────────
    print("\nCalibrating zero positions — keep the arm STILL...")
    zeros = {}
    for sid in found:
        angle = read_angle(ser, sid)
        zeros[sid] = angle if angle is not None else 0.0
        print(f"  ID {sid}: zero = {zeros[sid]:.1f}°")
    print("Calibration done.\n")

    # ── Step 3: Live display ──────────────────────────────────────────────────
    print("Live joint angles (degrees from zero).  Move the arm to verify each joint.")
    print("Press Ctrl+C to quit.\n")
    header = "  " + "  ".join(f"  J{i}   " for i in SERVO_IDS)
    print(header)
    print("  " + "-" * (N_SERVOS * 9))

    try:
        while True:
            parts = []
            for sid in SERVO_IDS:
                if sid in found:
                    angle = read_angle(ser, sid)
                    if angle is not None:
                        delta = angle - zeros[sid]
                        parts.append(f"{delta:+7.1f}")
                    else:
                        parts.append("  ERR  ")
                else:
                    parts.append("  ---  ")
            print("  " + "  ".join(parts), end="\r", flush=True)
            time.sleep(0.02)   # ~50 Hz
    except KeyboardInterrupt:
        print("\n\nCtrl+C — exiting.")
    finally:
        ser.close()
        print("Port closed.")


if __name__ == "__main__":
    main()
