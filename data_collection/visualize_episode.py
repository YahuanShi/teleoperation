#!/usr/bin/env python3
"""
Live visualization companion for UR5 teleoperation recording.

Run this alongside episode_recorder.py to get a richer, real-time view
of both cameras and the robot state/action streams.

Topics consumed (same as episode_recorder.py):
    /cam_1         sensor_msgs/Image       → exterior camera
    /cam_2         sensor_msgs/Image       → wrist camera
    /robot_state   Float64MultiArray (7,)  → actual joints [deg x6] + gripper [0-1]
    /robot_action  Float64MultiArray (7,)  → commanded action (same layout)

Usage:
    python3 teleoperation/data_collection/visualize_episode.py

Controls (focus the OpenCV window):
    Q / ESC     Quit
"""

from collections import deque
import threading
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

# ── Display constants ─────────────────────────────────────────────────────────
_CAM_W, _CAM_H = 480, 360  # display size of each camera panel
_INFO_H = 180  # height of the info panel below cameras
_PAD = 6  # gap between panels
_WIN_W = _CAM_W * 2 + _PAD * 3
_WIN_H = _CAM_H + _INFO_H + _PAD * 2

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GREEN = (100, 255, 100)
_YELLOW = (80, 230, 230)
_WHITE = (220, 220, 220)
_GRAY = (140, 140, 140)
_RED = (60, 60, 240)


# ── Thread-safe data buffer ───────────────────────────────────────────────────


class LiveBuffer:
    def __init__(self, node: Node):
        self._lock = threading.Lock()
        self._bridge = CvBridge()

        self._cam1_bgr: np.ndarray | None = None
        self._cam2_bgr: np.ndarray | None = None
        self._state: np.ndarray | None = None
        self._action: np.ndarray | None = None

        self._ts: deque = deque(maxlen=100)  # for Hz calculation

        node.create_subscription(Image, "/cam_1", self._cb_cam1, 10)
        node.create_subscription(Image, "/cam_2", self._cb_cam2, 10)
        node.create_subscription(Float64MultiArray, "/robot_state", self._cb_state, 10)
        node.create_subscription(Float64MultiArray, "/robot_action", self._cb_action, 10)

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _cb_cam1(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._lock:
                self._cam1_bgr = frame
        except Exception:
            pass

    def _cb_cam2(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._lock:
                self._cam2_bgr = frame
        except Exception:
            pass

    def _cb_state(self, msg: Float64MultiArray):
        with self._lock:
            self._state = np.array(msg.data, dtype=np.float64)
            self._ts.append(time.time())

    def _cb_action(self, msg: Float64MultiArray):
        with self._lock:
            self._action = np.array(msg.data, dtype=np.float64)

    # ── public API ────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        with self._lock:
            return all(x is not None for x in (self._cam1_bgr, self._cam2_bgr, self._state, self._action))

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "cam1": self._cam1_bgr.copy()
                if self._cam1_bgr is not None
                else np.zeros((_CAM_H, _CAM_W, 3), dtype=np.uint8),
                "cam2": self._cam2_bgr.copy()
                if self._cam2_bgr is not None
                else np.zeros((_CAM_H, _CAM_W, 3), dtype=np.uint8),
                "state": self._state.copy() if self._state is not None else None,
                "action": self._action.copy() if self._action is not None else None,
            }

    def hz(self) -> float:
        ts = list(self._ts)
        if len(ts) < 2:
            return 0.0
        dts = np.diff(ts)
        return float(1.0 / np.mean(dts)) if dts.size > 0 else 0.0


# ── Rendering helpers ─────────────────────────────────────────────────────────


def _put(img, text, pos, scale=0.48, color=_WHITE, thickness=1):
    cv2.putText(img, text, pos, _FONT, scale, color, thickness, cv2.LINE_AA)


def _prep_cam(bgr: np.ndarray, label: str) -> np.ndarray:
    out = cv2.resize(bgr, (_CAM_W, _CAM_H))
    _put(out, label, (10, 28), scale=0.7, color=_YELLOW, thickness=2)
    return out


def _build_info_panel(state: np.ndarray | None, action: np.ndarray | None, hz: float) -> np.ndarray:
    panel = np.zeros((_INFO_H, _WIN_W, 3), dtype=np.uint8)

    if state is None or action is None:
        _put(panel, "Waiting for /robot_state and /robot_action …", (_PAD, 40), color=_RED, scale=0.6)
        return panel

    # ── Hz bar ────────────────────────────────────────────────────────────
    max_hz = 30.0
    bx, by = _PAD, 14
    bw = _WIN_W - _PAD * 2
    bh = 8
    filled = int(bw * min(hz / max_hz, 1.0))
    cv2.rectangle(panel, (bx, by - bh // 2), (bx + bw, by + bh // 2), (50, 50, 50), -1)
    bar_color = _GREEN if hz >= 12 else (0, 165, 255) if hz >= 6 else _RED
    cv2.rectangle(panel, (bx, by - bh // 2), (bx + filled, by + bh // 2), bar_color, -1)
    _put(panel, f"State topic  {hz:.1f} Hz", (_PAD, 36), color=bar_color, scale=0.5)

    # ── Joint obs ─────────────────────────────────────────────────────────
    j_obs = "  ".join(f"J{i + 1}:{state[i]:+7.2f}°" for i in range(6))
    _put(panel, f"Obs  joints : {j_obs}", (_PAD, 62), color=_WHITE, scale=0.46)
    _put(panel, f"Obs  gripper: {state[6]:.3f}  (0=closed  1=open)", (_PAD, 84), color=_WHITE, scale=0.46)

    # ── Action ────────────────────────────────────────────────────────────
    j_act = "  ".join(f"J{i + 1}:{action[i]:+7.2f}°" for i in range(6))
    _put(panel, f"Cmd  joints : {j_act}", (_PAD, 110), color=_WHITE, scale=0.46)
    _put(panel, f"Cmd  gripper: {action[6]:.3f}", (_PAD, 132), color=_WHITE, scale=0.46)

    # ── Delta (tracking error) ────────────────────────────────────────────
    delta = action[:6] - state[:6]
    d_str = "  ".join(f"J{i + 1}:{delta[i]:+6.2f}°" for i in range(6))
    _put(panel, f"Delta       : {d_str}", (_PAD, 158), color=_GRAY, scale=0.43)

    # ── Hint ──────────────────────────────────────────────────────────────
    _put(panel, "[Q / ESC] quit", (_PAD, 176), scale=0.38, color=_GRAY)

    return panel


def build_frame(snap: dict, hz: float) -> np.ndarray:
    cam1 = _prep_cam(snap["cam1"], "Exterior")
    cam2 = _prep_cam(snap["cam2"], "Wrist")

    pad_v = np.zeros((_CAM_H, _PAD, 3), dtype=np.uint8)
    cameras = np.hstack([pad_v, cam1, pad_v, cam2, pad_v])

    info = _build_info_panel(snap["state"], snap["action"], hz)
    pad_h = np.zeros((_PAD, _WIN_W, 3), dtype=np.uint8)

    return np.vstack([cameras, pad_h, info])


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    rclpy.init()
    node = Node("visualize_episode_viewer")

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    buf = LiveBuffer(node)

    win = "Live View  [exterior | wrist]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, _WIN_W, _WIN_H)

    node.get_logger().info("[Viewer] Waiting for topics …")

    try:
        while rclpy.ok():
            snap = buf.snapshot()
            frame = build_frame(snap, buf.hz())
            cv2.imshow(win, frame)

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
