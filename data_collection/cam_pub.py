#!/usr/bin/env python3
"""
Dual RealSense Camera Publisher

Publishes:
    /cam_1  sensor_msgs/Image   exterior camera  (BGR8, 30 Hz)
    /cam_2  sensor_msgs/Image   wrist camera     (BGR8, 30 Hz)

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
SERIAL_1 = "105422060444"  # exterior camera (D415)
SERIAL_2 = "352122273671"  # wrist camera (D405)

PUBLISH_HZ = 30  # frames per second


class CameraNode(Node):
    def __init__(self):
        super().__init__("multi_cam_node")
        self._bridge = CvBridge()
        self._lock = threading.Lock()

        # Latest frames (protected by _lock)
        self._frame1: np.ndarray | None = None
        self._frame2: np.ndarray | None = None

        # ROS 2 publishers
        self._pub1 = self.create_publisher(Image, "/cam_1", 10)
        self._pub2 = self.create_publisher(Image, "/cam_2", 10)

        # Start pipelines
        self._pipe1 = self._start_pipeline(SERIAL_1)
        self._pipe2 = self._start_pipeline(SERIAL_2)

        # Per-camera capture threads (daemon so they die with the process)
        threading.Thread(target=self._capture_loop, args=(self._pipe1, 1), daemon=True, name="cam1-capture").start()
        threading.Thread(target=self._capture_loop, args=(self._pipe2, 2), daemon=True, name="cam2-capture").start()

        # Timer publishes both frames at PUBLISH_HZ
        self.create_timer(1.0 / PUBLISH_HZ, self._publish_frames)

        self.get_logger().info(f"[CamPub] Cameras started — publishing /cam_1 and /cam_2 at {PUBLISH_HZ} Hz.")

    # ── pipeline helpers ─────────────────────────────────────────────────────

    def _start_pipeline(self, serial: str) -> rs.pipeline:
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(cfg)
        self.get_logger().info(f"[CamPub] Pipeline started for serial {serial}")
        return pipeline

    # ── per-camera background capture threads ────────────────────────────────

    def _capture_loop(self, pipeline: rs.pipeline, cam_id: int):
        """Continuously grab frames from one camera into the shared buffer."""
        while rclpy.ok():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
                color = frames.get_color_frame()
                if color:
                    frame = np.asanyarray(color.get_data())
                    with self._lock:
                        if cam_id == 1:
                            self._frame1 = frame
                        else:
                            self._frame2 = frame
                else:
                    self.get_logger().warning(f"[CamPub] No color frame from camera {cam_id}")
            except Exception as e:
                self.get_logger().warning(f"[CamPub] Camera {cam_id} capture error: {e}", throttle_duration_sec=5.0)

    # ── publish timer callback ────────────────────────────────────────────────

    def _publish_frames(self):
        with self._lock:
            f1 = self._frame1
            f2 = self._frame2

        if f1 is not None:
            self._pub1.publish(self._bridge.cv2_to_imgmsg(f1, encoding="bgr8"))
        if f2 is not None:
            self._pub2.publish(self._bridge.cv2_to_imgmsg(f2, encoding="bgr8"))

    # ── clean shutdown ────────────────────────────────────────────────────────

    def destroy_node(self):
        """Stop camera pipelines before the node is torn down."""
        with contextlib.suppress(Exception):
            self._pipe1.stop()
        with contextlib.suppress(Exception):
            self._pipe2.stop()
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
