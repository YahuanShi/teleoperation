#!/usr/bin/env python3
"""
test_cam.py — Dual RealSense camera hardware test (no ROS required).

Shows both cameras side-by-side in an OpenCV window.
Press 'q' to quit.

Cameras:
    cam_1 (exterior) : Intel RealSense D415  serial 105422060444
    cam_2 (wrist)    : Intel RealSense D405  serial 352122273671
"""

import sys
import cv2
import numpy as np
import pyrealsense2 as rs

SERIAL_1 = "105422060444"  # exterior camera (D415)
SERIAL_2 = "352122273671"  # wrist camera (D405)


def start_pipeline(serial: str) -> rs.pipeline:
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)
    return pipeline


def main():
    print("Starting dual-camera test...")

    try:
        pipe1 = start_pipeline(SERIAL_1)
        print(f"[OK] cam_1 exterior D415 ({SERIAL_1})")
    except Exception as e:
        print(f"[FAIL] cam_1 D415 ({SERIAL_1}): {e}")
        sys.exit(1)

    try:
        pipe2 = start_pipeline(SERIAL_2)
        print(f"[OK] cam_2 wrist   D405 ({SERIAL_2})")
    except Exception as e:
        pipe1.stop()
        print(f"[FAIL] cam_2 D405 ({SERIAL_2}): {e}")
        sys.exit(1)

    print("Press 'Q' in the window to quit.\n")
    frame_count = 0

    try:
        while True:
            f1 = pipe1.wait_for_frames(timeout_ms=2000).get_color_frame()
            f2 = pipe2.wait_for_frames(timeout_ms=2000).get_color_frame()

            if not f1 or not f2:
                print("WARNING: missed frame")
                continue

            img1 = np.asanyarray(f1.get_data())
            img2 = np.asanyarray(f2.get_data())
            frame_count += 1

            cv2.putText(img1, f"cam_1 exterior D415  frame {frame_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img2, f"cam_2 wrist D405  frame {frame_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            combined = np.concatenate([img1, img2], axis=1)
            cv2.imshow("Camera Test  [cam_1 exterior | cam_2 wrist]  Q=quit", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipe1.stop()
        pipe2.stop()
        cv2.destroyAllWindows()
        print(f"[DONE] {frame_count} frames captured.")


if __name__ == "__main__":
    main()
