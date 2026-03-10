#!/usr/bin/env python3
"""
test_gripper.py — Weiss Robotics CRG 30-050 hardware test (no ROS required).

Tests the gripper over USB serial using the ASCII PDOUT/PDIN protocol
(DC-IOLINK firmware 1.0.7).  Runs open → close → open cycle and
reports measured jaw position at each step.

Usage:
    python3 test_gripper.py [port]
    python3 test_gripper.py /dev/ttyACM0   # default

Press Ctrl+C to abort at any time.
"""

import sys
import time
import serial

PORT = "/dev/ttyACM0"
BAUDRATE = 115200
PDIN_PER_MM = 163.17   # encoder units per mm (empirically measured)
CLOSED_PDIN = 150       # approximate PDIN at fully closed (calibrated by home())


def read_pdin(ser: serial.Serial, timeout: float = 1.0) -> int | None:
    """Read one fresh @PDIN push message; return raw PDIN value or None."""
    t0 = time.monotonic()
    ser.timeout = 0.2
    while time.monotonic() - t0 < timeout:
        line = ser.readline().decode("ascii", errors="ignore").strip()
        if line.startswith("@PDIN"):
            try:
                data = line[7:].split("]")[0].split(",")
                return (int(data[0], 16) << 8) | int(data[1], 16)
            except Exception:
                pass
    return None


def send(ser: serial.Serial, cmd: str) -> str:
    """Send ASCII command; skip @PDIN push lines; return first real response."""
    ser.reset_input_buffer()
    ser.write((cmd + "\n").encode("ascii"))
    ser.timeout = 1.0
    for _ in range(30):
        resp = ser.readline().decode("ascii", errors="ignore").strip()
        if resp.startswith("@PDIN"):
            continue
        return resp
    return ""


def wait_motion(ser: serial.Serial, timeout: float = 5.0) -> int:
    """
    Block until the gripper stops moving.
    Requires ≥200 PDIN change before checking stability (avoids false trigger).
    Returns final PDIN value.
    """
    start_pdin = None
    prev = None
    stable = 0
    moved = False
    t0 = time.monotonic()
    ser.timeout = 0.1
    while time.monotonic() - t0 < timeout:
        line = ser.readline().decode("ascii", errors="ignore").strip()
        if not line.startswith("@PDIN"):
            continue
        try:
            data = line[7:].split("]")[0].split(",")
            v = (int(data[0], 16) << 8) | int(data[1], 16)
        except Exception:
            continue
        if start_pdin is None:
            start_pdin = v
        if not moved and abs(v - start_pdin) >= 200:
            moved = True
        if moved:
            if prev is not None and abs(v - prev) <= 5:
                stable += 1
                if stable >= 15:
                    return v
            else:
                stable = 0
        prev = v
        mm = max(0.0, (v - CLOSED_PDIN) / PDIN_PER_MM)
        print(f"  PDIN={v:5d}  pos={mm:5.1f} mm", end="\r", flush=True)
    print()
    return prev or 0


def pdin_to_mm(pdin: int, closed_pdin: int) -> float:
    return max(0.0, (pdin - closed_pdin) / PDIN_PER_MM)


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else PORT
    print(f"Weiss CRG 30-050 test  —  port: {port}")
    print("=" * 48)

    try:
        ser = serial.Serial(port=port, baudrate=BAUDRATE, timeout=1.0)
    except Exception as e:
        print(f"[FAIL] Cannot open {port}: {e}")
        sys.exit(1)
    print(f"[OK]  Serial port {port} opened @ {BAUDRATE} baud")

    time.sleep(0.3)

    # ── Step 1: Read current position ────────────────────────────────────────
    print("\n[1/4] Reading current position...")
    pdin = read_pdin(ser)
    if pdin is None:
        print("[FAIL] No @PDIN stream received — check USB cable and 24 V power.")
        ser.close()
        sys.exit(1)
    mm = pdin_to_mm(pdin, CLOSED_PDIN)
    print(f"      PDIN={pdin}  ≈ {mm:.1f} mm  (raw, before calibration)")

    # ── Step 2: Calibrate closed position (home) ─────────────────────────────
    print("\n[2/4] Calibrating — closing to find closed_pdin...")
    print("      (gripper will move — ensure jaws are clear)")
    input("      Press Enter to continue, Ctrl+C to abort: ")

    send(ser, "PDOUT=[00,00]")
    time.sleep(0.1)
    send(ser, "PDOUT=[03,00]")
    closed_pdin = wait_motion(ser, timeout=5.0)
    print(f"\n      closed_pdin = {closed_pdin}")

    # ── Step 3: Open fully ───────────────────────────────────────────────────
    print("\n[3/4] Opening gripper (PDOUT=[07,00])...")
    send(ser, "PDOUT=[00,00]")
    time.sleep(0.1)
    send(ser, "PDOUT=[07,00]")
    open_pdin = wait_motion(ser, timeout=5.0)
    open_mm = pdin_to_mm(open_pdin, closed_pdin)
    print(f"\n      open_pdin = {open_pdin}  →  {open_mm:.1f} mm")

    # ── Step 4: Close again ──────────────────────────────────────────────────
    print("\n[4/4] Closing gripper (PDOUT=[03,00])...")
    send(ser, "PDOUT=[00,00]")
    time.sleep(0.1)
    send(ser, "PDOUT=[03,00]")
    final_pdin = wait_motion(ser, timeout=5.0)
    final_mm = pdin_to_mm(final_pdin, closed_pdin)
    print(f"\n      final_pdin = {final_pdin}  →  {final_mm:.1f} mm")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 48)
    print("RESULT")
    print(f"  closed_pdin : {closed_pdin}")
    print(f"  open_pdin   : {open_pdin}")
    print(f"  stroke      : {open_pdin - closed_pdin} PDIN  ≈  {open_mm:.1f} mm")
    if open_mm >= 25.0:
        print("[PASS] Full stroke OK")
    else:
        print(f"[WARN] Stroke only {open_mm:.1f} mm (expected ~30 mm) — check jaws")

    send(ser, "PDOUT=[00,00]")
    ser.close()
    print("Serial port closed.")


if __name__ == "__main__":
    main()
