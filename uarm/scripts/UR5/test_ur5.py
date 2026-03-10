#!/usr/bin/env python3
"""
test_ur5.py — UR5 connectivity and basic motion test (no ROS required).

Steps:
    1. Connect RTDE receive — read actual joint angles
    2. Connect RTDE control — verify control interface
    3. Move to home position (safe, slow)
    4. Print live joint angles for 5 seconds

Usage:
    python3 test_ur5.py [ip]
    python3 test_ur5.py 192.168.1.98   # default — match servo2ur5.py UR5_IP

Press Ctrl+C to abort at any time.
"""

import sys
import time
import numpy as np

try:
    import rtde_control
    import rtde_receive
except ImportError:
    print("[FAIL] ur_rtde not installed. Run: pip install ur-rtde")
    sys.exit(1)

# Must match UR5_IP in servo2ur5.py AND ur5_pub.py
UR5_IP = "192.168.1.98"

# Safe home position — matches servo2ur5.py UR5_HOME_DEG
HOME_DEG = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]


def main():
    ip = sys.argv[1] if len(sys.argv) > 1 else UR5_IP
    print(f"UR5 connectivity test  —  IP: {ip}")
    print("=" * 48)

    # ── Step 1: RTDE receive ─────────────────────────────────────────────────
    print("\n[1/4] Connecting RTDE receive...")
    try:
        rtde_r = rtde_receive.RTDEReceiveInterface(ip)
        print(f"[OK]  RTDE receive connected to {ip}")
    except Exception as e:
        print(f"[FAIL] Cannot connect to {ip}: {e}")
        print("       Check: UR5 powered on, Ethernet cable, correct IP.")
        sys.exit(1)

    # ── Step 2: Read current joints ──────────────────────────────────────────
    print("\n[2/4] Reading current joint angles...")
    try:
        q_rad = rtde_r.getActualQ()
        q_deg = np.degrees(q_rad)
        print(f"[OK]  Current joints (deg):")
        for name, deg in zip(JOINT_NAMES, q_deg):
            print(f"        {name:15s}: {deg:+7.2f}°")
    except Exception as e:
        print(f"[FAIL] Cannot read joints: {e}")
        sys.exit(1)

    # ── Step 3: RTDE control ─────────────────────────────────────────────────
    print("\n[3/4] Connecting RTDE control...")
    try:
        rtde_c = rtde_control.RTDEControlInterface(ip)
        print(f"[OK]  RTDE control connected.")
    except Exception as e:
        print(f"[FAIL] Cannot open control interface: {e}")
        print("       Ensure robot is not in remote-control lockout (check teach pendant).")
        sys.exit(1)

    # ── Step 4: Move to home ─────────────────────────────────────────────────
    print(f"\n[4/4] Moving to home position: {HOME_DEG} deg")
    print("      (slow moveJ — ensure workspace is clear)")
    input("      Press Enter to move, Ctrl+C to abort: ")

    try:
        home_rad = np.radians(HOME_DEG).tolist()
        rtde_c.moveJ(home_rad, speed=0.3, acceleration=0.3)
        print("[OK]  Reached home position.")
    except Exception as e:
        print(f"[FAIL] moveJ error: {e}")
        rtde_c.stopJ(2.0)
        sys.exit(1)

    # ── Live read for 5 s ────────────────────────────────────────────────────
    print("\nLive joint angles for 5 seconds...")
    header = "  " + "  ".join(f"{n[:8]:>9}" for n in JOINT_NAMES)
    print(header)
    t0 = time.time()
    try:
        while time.time() - t0 < 5.0:
            q = np.degrees(rtde_r.getActualQ())
            row = "  " + "  ".join(f"{v:+9.2f}" for v in q)
            print(row, end="\r", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    print()

    rtde_c.stopScript()
    print("\n[PASS] UR5 test complete.")


if __name__ == "__main__":
    main()
