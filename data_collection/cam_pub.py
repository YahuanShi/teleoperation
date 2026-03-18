#!/usr/bin/env python3
"""
Adaptive RealSense Camera Publisher

Publishes:
    /cam_1  sensor_msgs/Image   exterior camera  (BGR8, 30 Hz)  [required]
    /cam_2  sensor_msgs/Image   wrist camera     (BGR8, 30 Hz)  [required]
    /cam_3  sensor_msgs/Image   front camera     (BGR8, 30 Hz)  [optional — only if connected]

Each camera runs its own background thread so a stall on one camera
does not delay the other.  Cleanup is done via destroy_node() rather
than __del__ so the pipeline stop is guaranteed on shutdown.
"""

import contextlib
import threading

import cv2
from cv_bridge import CvBridge
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# ── Camera serial numbers ──────────────
SERIAL_1 = "105422061000"  # exterior camera (D415)  [required]
SERIAL_2 = "352122273671"  # wrist camera (D405)     [required]
SERIAL_3 = "104122061227"  # front camera (D415, USB 2.1) [optional]

PUBLISH_HZ = 30  # frames per second


def _connected_serials() -> set[str]:
    """Return the set of RealSense serial numbers visible to pyrealsense2."""
    ctx = rs.context()
    return {d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()}


class CameraNode(Node):
    def __init__(self):
        super().__init__("multi_cam_node")
        self._bridge = CvBridge()
        self._lock = threading.Lock()

        connected = _connected_serials()
        self._has_front = SERIAL_3 in connected
        self.get_logger().info(
            f"[CamPub] Detected serials: {connected}  →  "
            f"{'3-camera' if self._has_front else '2-camera'} mode"
        )

        # Latest frames (protected by _lock)
        self._frame1: np.ndarray | None = None
        self._frame2: np.ndarray | None = None
        self._frame3: np.ndarray | None = None

        # ROS 2 publishers (cam_3 only if front camera is connected)
        self._pub1 = self.create_publisher(Image, "/cam_1", 10)
        self._pub2 = self.create_publisher(Image, "/cam_2", 10)
        self._pub3 = self.create_publisher(Image, "/cam_3", 10) if self._has_front else None

        # Start required pipelines
        self._pipe1 = self._start_pipeline(SERIAL_1)
        self._pipe2 = self._start_pipeline(SERIAL_2)
        self._pipe3 = self._start_pipeline(SERIAL_3) if self._has_front else None

        # Per-camera capture threads (daemon so they die with the process)
        threading.Thread(target=self._capture_loop, args=(self._pipe1, 1), daemon=True, name="cam1-capture").start()
        threading.Thread(target=self._capture_loop, args=(self._pipe2, 2), daemon=True, name="cam2-capture").start()
        if self._has_front:
            threading.Thread(target=self._capture_loop, args=(self._pipe3, 3), daemon=True, name="cam3-capture").start()

        # Timer publishes all frames at PUBLISH_HZ
        self.create_timer(1.0 / PUBLISH_HZ, self._publish_frames)

        topics = "/cam_1, /cam_2" + (", /cam_3" if self._has_front else "")
        self.get_logger().info(f"[CamPub] Publishing {topics} at {PUBLISH_HZ} Hz.")

    # ── pipeline helpers ─────────────────────────────────────────────────────

    def _start_pipeline(self, serial: str) -> rs.pipeline | None:
        """Try to start a RealSense pipeline; return None on failure."""
        # Format fallback list: BGR8/RGB8 first, then YUYV (USB 2.1 cameras)
        attempts = [
            (640, 480, rs.format.bgr8, 30),
            (848, 480, rs.format.bgr8, 30),
            (640, 480, rs.format.rgb8, 30),
            (640, 480, rs.format.yuyv, 30),   # USB 2.1 fallback
            (640, 480, rs.format.yuyv, 15),   # USB 2.1 low-bandwidth fallback
        ]
        for w, h, fmt, fps in attempts:
            try:
                pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_device(serial)
                cfg.enable_stream(rs.stream.color, w, h, fmt, fps)
                pipeline.start(cfg)
                self.get_logger().info(
                    f"[CamPub] Pipeline started for serial {serial} ({w}x{h} {fmt} {fps}fps)"
                )
                return pipeline
            except Exception as e:
                self.get_logger().warning(
                    f"[CamPub] serial {serial} failed with ({w}x{h} {fmt}): {e}"
                )
        self.get_logger().error(
            f"[CamPub] All attempts failed for serial {serial} — publishing black frames."
        )
        return None

    # ── per-camera background capture threads ────────────────────────────────

    _BLANK = np.zeros((480, 640, 3), dtype=np.uint8)

    def _capture_loop(self, pipeline: rs.pipeline | None, cam_id: int):
        """Continuously grab frames from one camera into the shared buffer."""
        if pipeline is None:
            # Publish a black frame immediately so the recorder doesn't stall.
            with self._lock:
                if cam_id == 1:
                    self._frame1 = self._BLANK.copy()
                elif cam_id == 2:
                    self._frame2 = self._BLANK.copy()
                else:
                    self._frame3 = self._BLANK.copy()
            return  # thread exits; blank frame stays in buffer
        while rclpy.ok():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
                color = frames.get_color_frame()
                if color:
                    frame = np.asanyarray(color.get_data())
                    # YUYV (from USB 2.1 cameras) → BGR8
                    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 2):
                        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
                    with self._lock:
                        if cam_id == 1:
                            self._frame1 = frame
                        elif cam_id == 2:
                            self._frame2 = frame
                        else:
                            self._frame3 = frame
                else:
                    self.get_logger().warning(f"[CamPub] No color frame from camera {cam_id}")
            except Exception as e:
                self.get_logger().warning(f"[CamPub] Camera {cam_id} capture error: {e}", throttle_duration_sec=5.0)

    # ── publish timer callback ────────────────────────────────────────────────

    def _publish_frames(self):
        with self._lock:
            f1 = self._frame1
            f2 = self._frame2
            f3 = self._frame3 if self._has_front else None

        if f1 is not None:
            self._pub1.publish(self._bridge.cv2_to_imgmsg(f1, encoding="bgr8"))
        if f2 is not None:
            self._pub2.publish(self._bridge.cv2_to_imgmsg(f2, encoding="bgr8"))
        if f3 is not None and self._pub3 is not None:
            self._pub3.publish(self._bridge.cv2_to_imgmsg(f3, encoding="bgr8"))

    # ── clean shutdown ────────────────────────────────────────────────────────

    def destroy_node(self):
        """Stop camera pipelines before the node is torn down."""
        pipes = [self._pipe1, self._pipe2]
        if self._has_front:
            pipes.append(self._pipe3)
        for pipe in pipes:
            if pipe is not None:
                with contextlib.suppress(Exception):
                    pipe.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
