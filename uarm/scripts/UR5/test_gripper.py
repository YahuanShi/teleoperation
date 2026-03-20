#!/usr/bin/env python3
"""
test_gripper.py — Weiss IEG76 / CRG gripper hardware test (no ROS required).

Protocol reference: https://github.com/ipa320/weiss_gripper_ieg76

Serial: /dev/ttyACM0 @ 9600 baud (DC-IOLink USB adapter)

@PDIN=[B0,B1,B2,B3]  push message format:
    position_mm = ((B0 << 8) | B1) / 100.0
    B3 status flags:
        bit0 IDLE    bit1 OPEN    bit2 CLOSED
        bit3 HOLDING bit4 FAULT  bit5 TEMPFAULT
        bit6 TEMPWARN bit7 MAINT

Initialisation MUST follow this order before any PDOUT command:
    ID? → FALLBACK(1) → MODE? → RESTART() → OPERATE() → PDOUT=[00,00]

Usage:
    python3 test_gripper.py [port]
    python3 test_gripper.py /dev/ttyACM0   # default

Press Ctrl+C to abort at any time.
"""

import sys
import time

import serial

PORT = "/dev/ttyACM0"
BAUDRATE = 9600

# Status flag bit positions in B3
FLAG_IDLE = 0
FLAG_OPEN = 1
FLAG_CLOSED = 2
FLAG_HOLDING = 3
FLAG_FAULT = 4
FLAG_TEMPFAULT = 5
FLAG_TEMPWARN = 6
FLAG_MAINT = 7

# Default positions
OPEN_MM = 29.5
CLOSE_MM = 0.5


def _pos_to_bytes(mm: float) -> str:
    """Encode position mm → '[B0,B1]' hex string for SETPARAM."""
    val = int(mm * 100)
    b0 = (val >> 8) & 0xFF
    b1 = val & 0xFF
    return f"[{b0:02x},{b1:02x}]"


def send(ser: serial.Serial, cmd: str, wait: float = 0.3) -> None:
    ser.reset_input_buffer()
    ser.write((cmd + "\n").encode("ascii"))
    time.sleep(wait)


def read_pdin(ser: serial.Serial, timeout: float = 2.0) -> tuple[float, int] | tuple[None, None]:
    """Read one @PDIN push message. Returns (position_mm, flags_byte) or (None, None)."""
    t0 = time.monotonic()
    ser.timeout = 0.2
    while time.monotonic() - t0 < timeout:
        line = ser.readline().decode("ascii", errors="ignore").strip()
        if line.startswith("@PDIN=["):
            try:
                inner = line[7:].split("]")[0]
                parts = [int(x, 16) for x in inner.split(",")]
                pos_mm = ((parts[0] << 8) | parts[1]) / 100.0
                flags = parts[3] if len(parts) >= 4 else 0
                return pos_mm, flags
            except Exception:
                pass
    return None, None


def flag_str(flags: int) -> str:
    names = ["IDLE", "OPEN", "CLOSED", "HOLDING", "FAULT", "TEMPFAULT", "TEMPWARN", "MAINT"]
    return " | ".join(n for i, n in enumerate(names) if flags & (1 << i)) or "none"


def wait_for_flag(ser: serial.Serial, flag_bit: int, timeout: float = 6.0) -> bool:
    """Wait until a specific status flag is set."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        pos_mm, flags = read_pdin(ser, timeout=0.5)
        if flags is not None:
            print(f"  pos={pos_mm:5.2f} mm  flags=[{flag_str(flags)}]", end="\r", flush=True)
            if flags & (1 << flag_bit):
                print()
                return True
    print()
    return False


def initialise(ser: serial.Serial) -> bool:
    """Run the mandatory startup sequence."""
    print("  Sending: ID?")
    send(ser, "ID?", 0.5)
    send(ser, "ID?", 0.5)
    print("  Sending: FALLBACK(1)")
    send(ser, "FALLBACK(1)", 0.5)
    print("  Sending: MODE?")
    send(ser, "MODE?", 0.5)
    print("  Sending: RESTART()")
    send(ser, "RESTART()", 0.5)
    print("  Sending: OPERATE()")
    send(ser, "OPERATE()", 0.5)
    print("  Sending: PDOUT=[00,00] (reset)")
    send(ser, "PDOUT=[00,00]", 0.5)

    pos, flags = read_pdin(ser, timeout=2.0)
    if pos is None:
        return False
    print(f"  [OK] Gripper responding — pos={pos:.2f} mm  flags=[{flag_str(flags)}]")
    return True


def set_positions(ser: serial.Serial, open_mm: float, close_mm: float) -> None:
    open_hex = _pos_to_bytes(open_mm)
    close_hex = _pos_to_bytes(close_mm)
    send(ser, f"SETPARAM(96, 2, {open_hex})", 0.3)  # opening position
    send(ser, f"SETPARAM(96, 1, {close_hex})", 0.3)  # closing position
    send(ser, "SETPARAM(96, 3, [64])", 0.3)  # 100% force


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else PORT
    print(f"Weiss gripper test  —  port: {port}  @ {BAUDRATE} baud")
    print("=" * 52)

    try:
        ser = serial.Serial(port=port, baudrate=BAUDRATE, timeout=0.2)
    except Exception as e:
        print(f"[FAIL] Cannot open {port}: {e}")
        sys.exit(1)
    print("[OK]  Port opened.")
    time.sleep(0.5)

    # ── Step 1: Initialise ───────────────────────────────────────────────────
    print("\n[1/5] Initialising gripper...")
    if not initialise(ser):
        print("[FAIL] No @PDIN response after init — check 24V and IO-Link cable.")
        ser.close()
        sys.exit(1)

    # ── Step 2: Reference (home) ─────────────────────────────────────────────
    print("\n[2/5] Referencing gripper (home cycle — jaws will close then open)...")
    input("      Press Enter to start, Ctrl+C to abort: ")

    set_positions(ser, OPEN_MM, CLOSE_MM)
    send(ser, "PDOUT=[07,00]", 0.2)  # reference command
    if not wait_for_flag(ser, FLAG_OPEN, timeout=10.0):
        # Gripper may already be open — check current flags
        pos, flags = read_pdin(ser, timeout=1.0)
        if flags is not None and (flags & (1 << FLAG_OPEN) or flags & (1 << FLAG_IDLE)):
            print("[OK]  Already at open position — reference skipped.")
        else:
            print("[WARN] Reference did not complete within 10 s — continuing anyway.")

    pos, flags = read_pdin(ser, timeout=1.0)
    print(f"[OK]  After reference: pos={pos:.2f} mm  flags=[{flag_str(flags)}]")

    # ── Step 3: Close ────────────────────────────────────────────────────────
    print("\n[3/5] Closing gripper...")
    set_positions(ser, OPEN_MM, CLOSE_MM)
    send(ser, "PDOUT=[03,00]", 0.2)
    if not wait_for_flag(ser, FLAG_CLOSED, timeout=6.0):
        # HOLDING is also acceptable (grasped something)
        pos, flags = read_pdin(ser, timeout=0.5)
        if flags and (flags & (1 << FLAG_HOLDING)):
            print("[OK]  Gripper is HOLDING (object in jaws).")
        else:
            print("[WARN] CLOSED flag not set — check if jaws are blocked.")
    pos, flags = read_pdin(ser, timeout=1.0)
    print(f"[OK]  After close: pos={pos:.2f} mm  flags=[{flag_str(flags)}]")

    # ── Step 4: Open ─────────────────────────────────────────────────────────
    print("\n[4/5] Opening gripper...")
    set_positions(ser, OPEN_MM, CLOSE_MM)
    send(ser, "PDOUT=[02,00]", 0.2)
    if not wait_for_flag(ser, FLAG_OPEN, timeout=6.0):
        print("[WARN] OPEN flag not set.")
    pos, flags = read_pdin(ser, timeout=1.0)
    print(f"[OK]  After open: pos={pos:.2f} mm  flags=[{flag_str(flags)}]")

    # ── Step 5: Deactivate ───────────────────────────────────────────────────
    print("\n[5/5] Deactivating...")
    send(ser, "PDOUT=[00,00]", 0.3)
    send(ser, "FALLBACK(1)", 0.3)
    pos, flags = read_pdin(ser, timeout=1.0)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("RESULT")
    if flags is not None and not (flags & (1 << FLAG_FAULT)):
        print(f"  Final pos : {pos:.2f} mm")
        print(f"  Flags     : {flag_str(flags)}")
        print("[PASS] Gripper test complete — no fault.")
    else:
        print(f"  Flags     : {flag_str(flags)}")
        print("[FAIL] Fault flag set — check gripper.")

    ser.close()
    print("Port closed.")


if __name__ == "__main__":
    main()
