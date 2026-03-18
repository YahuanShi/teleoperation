#!/usr/bin/env python3
"""
test_cam.py — RealSense camera hardware test (no ROS required).

Shows all connected cameras side-by-side in an OpenCV window.
Press 'q' to quit.

Cameras:
    cam_1 (exterior) : Intel RealSense D415  serial 105422061000  [required]
    cam_2 (wrist)    : Intel RealSense D405  serial 352122273671  [required]
    cam_3 (front)    : Intel RealSense D415  serial 104122061227  [optional — USB 2.1]
"""

import sys

import cv2
import numpy as np
import pyrealsense2 as rs

SERIAL_1 = "105422061000"  # exterior camera (D415)  [required]
SERIAL_2 = "352122273671"  # wrist camera (D405)     [required]
SERIAL_3 = "104122061227"  # front camera (D415, USB 2.1) [optional]

# Format fallback list — USB 2.1 cameras only support YUYV
_FORMAT_ATTEMPTS = [
    (640, 480, rs.format.bgr8, 30),
    (640, 480, rs.format.rgb8, 30),
    (640, 480, rs.format.yuyv, 30),
    (640, 480, rs.format.yuyv, 15),
]


def _connected_serials() -> set:
    ctx = rs.context()
    return {d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()}


def start_pipeline(serial: str) -> rs.pipeline:
    for w, h, fmt, fps in _FORMAT_ATTEMPTS:
        try:
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, w, h, fmt, fps)
            pipeline.start(cfg)
            print(f"       ({w}x{h} {fmt} {fps}fps)")
            return pipeline
        except Exception:
            pass
    raise RuntimeError(f"No supported format found for serial {serial}")


def grab_bgr(pipeline: rs.pipeline) -> np.ndarray | None:
    frames = pipeline.wait_for_frames(timeout_ms=2000)
    color = frames.get_color_frame()
    if not color:
        return None
    frame = np.asanyarray(color.get_data())
    # YUYV → BGR
    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 2):
        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
    return frame


def main():
    connected = _connected_serials()
    has_front = SERIAL_3 in connected
    print(f"Detected serials: {connected}")
    print(f"Starting {'3' if has_front else '2'}-camera test...")

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

    pipe3 = None
    if has_front:
        try:
            pipe3 = start_pipeline(SERIAL_3)
            print(f"[OK] cam_3 front   D415 ({SERIAL_3})")
        except Exception as e:
            print(f"[WARN] cam_3 D415 ({SERIAL_3}): {e} — continuing without front camera")
            has_front = False

    cam_labels = ["cam_1 exterior D415", "cam_2 wrist D405"]
    if has_front:
        cam_labels.append("cam_3 front D415")

    title = "Camera Test  [" + " | ".join(cam_labels) + "]  Q=quit"
    print(f"Press 'Q' in the window to quit.\n")
    frame_count = 0

    try:
        while True:
            img1 = grab_bgr(pipe1)
            img2 = grab_bgr(pipe2)
            img3 = grab_bgr(pipe3) if pipe3 else None

            if img1 is None or img2 is None:
                print("WARNING: missed frame")
                continue

            frame_count += 1

            for img, label, color in [
                (img1, f"cam_1 exterior D415  frame {frame_count}", (0, 255, 0)),
                (img2, f"cam_2 wrist D405  frame {frame_count}",    (0, 255, 255)),
            ]:
                cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            panels = [img1, img2]
            if img3 is not None:
                cv2.putText(img3, f"cam_3 front D415  frame {frame_count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                panels.append(img3)

            combined = np.concatenate(panels, axis=1)
            cv2.imshow(title, combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipe1.stop()
        pipe2.stop()
        if pipe3:
            pipe3.stop()
        cv2.destroyAllWindows()
        print(f"[DONE] {frame_count} frames captured.")


if __name__ == "__main__":
    main()
