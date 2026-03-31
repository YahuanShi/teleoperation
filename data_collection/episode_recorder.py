#!/usr/bin/env python3
"""
Episode Recorder
======================

General-purpose teleoperation episode recorder.  Subscribes to ROS 2 topics
published by any robot teleoperation stack and records demonstration episodes
in HDF5 format compatible with openpi / LeRobot training (including pi0.5).

Topics consumed:
    /cam_1         sensor_msgs/Image       → exterior camera        [required]
    /cam_2         sensor_msgs/Image       → wrist / gripper camera [required]
    /cam_3         sensor_msgs/Image       → front camera           [optional — auto-detected]
    /robot_state   Float64MultiArray (7,)  → actual joints [deg x6] + gripper [0-1]
    /robot_action  Float64MultiArray (7,)  → commanded action (same layout)

Camera count is auto-detected at startup: if /cam_3 messages arrive within 5 s,
the recorder runs in 3-camera mode; otherwise 2-camera mode (exterior + wrist only).
The wrist (cam_2) and exterior (cam_1) cameras are always required.

HDF5 layout written per episode (aloha-compatible, pi0.5 ready):
    /observations/images/exterior_image_1_left  (T, 480, 480, 3) uint8  RGB
    /observations/images/wrist_image_left        (T, 480, 480, 3) uint8  RGB
    /observations/images/front_image_1           (T, 480, 480, 3) uint8  RGB  [3-cam mode only]
    /observations/qpos                           (T, 7) float64
    /action                                      (T, 7) float64
    root.attrs: sim, prompt, task, hz, timestamp, n_steps, num_cameras

Keyboard shortcuts (focus the OpenCV preview window):
    P  →  Set task prompt      type once, reused for every episode in this session
    B  →  Begin recording      requires prompt to be set first
    S  →  Stop & Save          stops recording and immediately writes HDF5 to disk
    D  →  Delete last episode  removes the previously saved HDF5 file (undo save)
    Q  →  Quit                 safely exits (Ctrl-C also works)

Usage:
    python3 episode_recorder.py
    python3 episode_recorder.py --task pick_place
    python3 episode_recorder.py --task pick_place --data-dir ~/my_data --hz 30
    python3 episode_recorder.py --task pick_place --episode-idx 5
"""

import argparse
from collections import deque
import contextlib
import datetime
import os
import queue
import signal
import sys
import threading
import time

import cv2
from cv_bridge import CvBridge
import h5py
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray

# Global keyboard listener (works regardless of which window has focus).
# Falls back gracefully if pynput is unavailable.
_key_queue: queue.SimpleQueue = queue.SimpleQueue()
try:
    from pynput import keyboard as _kb

    def _on_press(key):
        with contextlib.suppress(AttributeError):
            _key_queue.put(key.char)  # special key (shift, ctrl, …) has no .char — ignore

    _kb.Listener(on_press=_on_press, daemon=True).start()
except Exception:
    pass  # pynput unavailable — cv2.waitKey only

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Resolve openpi project root: this script lives at
#   <openpi_root>/teleoperation/data_collection/episode_recorder.py
_OPENPI_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_TODAY = datetime.datetime.now(tz=datetime.timezone.utc).date().strftime("%Y%m%d")

TASK_CONFIGS: dict = {
    "default": {
        "dataset_dir": os.path.join(_OPENPI_ROOT, "dataset", f"ur5_dataset_{_TODAY}"),
        "episode_len": 100000,
        "hz": 16,
    },
    # "pick_place": {
    #     "dataset_dir": os.path.join(_OPENPI_ROOT, "dataset", f"ur5_pick_place_{_TODAY}"),
    #     "episode_len": 300,
    #     "hz": 10,
    # },
}

IMAGE_H, IMAGE_W = 224, 224
PREVIEW_H, PREVIEW_W = 224, 224  # per-camera frame size
HEADER_H = 28  # camera label strip above images
STATUS_H = 88  # info panel below images

# ═══════════════════════════════════════════════════════════════════════════════
# Image preprocessing
# ═══════════════════════════════════════════════════════════════════════════════


def center_crop_resize(image: np.ndarray, out_h: int = IMAGE_H, out_w: int = IMAGE_W) -> np.ndarray:
    h, w = image.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    cropped = image[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)


def resize_with_pad(image: np.ndarray, out_h: int = IMAGE_H, out_w: int = IMAGE_W) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(out_h / h, out_w / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    y0, x0 = (out_h - nh) // 2, (out_w - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


# Active preprocessing — swap to resize_with_pad if preferred.
preprocess_image = center_crop_resize

# ═══════════════════════════════════════════════════════════════════════════════
# Thread-safe ROS 2 data buffer
# ═══════════════════════════════════════════════════════════════════════════════


class DataBuffer:
    """
    Creates subscriptions on an existing Node and caches the most recent
    sample of each topic.  Thread-safe.

    Automatically detects whether /cam_3 (front camera) is available:
    waits up to 5 s for it after the required topics are ready, then
    locks in 2-camera or 3-camera mode via ``has_front_camera``.
    """

    def __init__(self, node: Node):
        self._node = node
        self._lock = threading.Lock()
        self._bridge = CvBridge()

        self._cam1_bgr: np.ndarray | None = None
        self._cam2_bgr: np.ndarray | None = None
        self._cam3_bgr: np.ndarray | None = None
        self._state: np.ndarray | None = None
        self._action: np.ndarray | None = None

        self._step_ts: deque = deque(maxlen=300)
        self._last_warn: dict[str, float] = {}

        node.create_subscription(Image, "/cam_1", self._cb_cam1, 2)
        node.create_subscription(Image, "/cam_2", self._cb_cam2, 2)
        node.create_subscription(Image, "/cam_3", self._cb_cam3, 2)
        node.create_subscription(Float64MultiArray, "/robot_state", self._cb_state, 10)
        node.create_subscription(Float64MultiArray, "/robot_action", self._cb_action, 10)

        # ── Wait for the required topics (cam1, cam2, state, action) ─────────
        node.get_logger().info("[Recorder] Waiting for required topics (cam_1, cam_2, robot_state, robot_action) …")
        deadline = time.time() + 30.0
        while not self._core_ready():
            if not rclpy.ok():
                raise RuntimeError("ROS shutdown while waiting for topics")
            if time.time() > deadline:
                raise RuntimeError("Required topics not received after 30 s — ensure teleoperation nodes are running.")
            time.sleep(0.1)

        # ── Check for optional front camera (cam_3) with a short timeout ─────
        node.get_logger().info("[Recorder] Required topics ready. Checking for optional /cam_3 (5 s) …")
        cam3_deadline = time.time() + 5.0
        while time.time() < cam3_deadline and rclpy.ok():
            with self._lock:
                if self._cam3_bgr is not None:
                    break
            time.sleep(0.1)

        with self._lock:
            self.has_front_camera: bool = self._cam3_bgr is not None

        cam_count = 3 if self.has_front_camera else 2
        node.get_logger().info(
            f"[Recorder] {'Front camera detected' if self.has_front_camera else 'No front camera'} "
            f"— running in {cam_count}-camera mode."
        )

    # ── throttled warning helper ──────────────────────────────────────────────

    def _warn_throttle(self, key: str, msg: str, interval: float = 5.0):
        now = time.time()
        if now - self._last_warn.get(key, 0.0) >= interval:
            self._node.get_logger().warning(msg)
            self._last_warn[key] = now

    # ── callbacks ────────────────────────────────────────────────────────────

    def _cb_cam1(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._lock:
                self._cam1_bgr = frame
        except Exception as e:
            self._warn_throttle("cam1", f"[Recorder] cam_1 error: {e}")

    def _cb_cam2(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._lock:
                self._cam2_bgr = frame
        except Exception as e:
            self._warn_throttle("cam2", f"[Recorder] cam_2 error: {e}")

    def _cb_cam3(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._lock:
                self._cam3_bgr = frame
        except Exception as e:
            self._warn_throttle("cam3", f"[Recorder] cam_3 error: {e}")

    def _cb_state(self, msg: Float64MultiArray):
        with self._lock:
            self._state = np.array(msg.data, dtype=np.float64)

    def _cb_action(self, msg: Float64MultiArray):
        with self._lock:
            self._action = np.array(msg.data, dtype=np.float64)

    # ── public API ────────────────────────────────────────────────────────────

    def _core_ready(self) -> bool:
        with self._lock:
            return all(x is not None for x in [self._cam1_bgr, self._cam2_bgr, self._state, self._action])

    def is_ready(self) -> bool:
        with self._lock:
            core = all(x is not None for x in [self._cam1_bgr, self._cam2_bgr, self._state, self._action])
            if self.has_front_camera:
                return core and self._cam3_bgr is not None
            return core

    def get_snapshot(self) -> dict | None:
        with self._lock:
            # Inline check — do NOT call is_ready() here (it also acquires _lock → deadlock)
            required = [self._cam1_bgr, self._cam2_bgr, self._state, self._action]
            if self.has_front_camera:
                required.append(self._cam3_bgr)
            if any(x is None for x in required):
                return None
            snap = {
                "cam1_rgb": cv2.cvtColor(self._cam1_bgr, cv2.COLOR_BGR2RGB),
                "cam2_rgb": cv2.cvtColor(self._cam2_bgr, cv2.COLOR_BGR2RGB),
                "state": self._state.copy(),
                "action": self._action.copy(),
            }
            if self.has_front_camera:
                snap["cam3_rgb"] = cv2.cvtColor(self._cam3_bgr, cv2.COLOR_BGR2RGB)
        self._step_ts.append(time.time())
        return snap

    def get_preview_frames(self) -> list:
        """Return list of [cam1, cam2] or [cam1, cam2, cam3] BGR frames."""
        with self._lock:
            c1 = self._cam1_bgr.copy() if self._cam1_bgr is not None else None
            c2 = self._cam2_bgr.copy() if self._cam2_bgr is not None else None
            frames = [c1, c2]
            if self.has_front_camera:
                c3 = self._cam3_bgr.copy() if self._cam3_bgr is not None else None
                frames.append(c3)
        return frames

    def reset_hz_counter(self) -> None:
        """Clear the step-timestamp history so avg_hz() starts fresh for a new episode."""
        self._step_ts.clear()

    def avg_hz(self) -> float:
        ts = list(self._step_ts)
        if len(ts) < 2:
            return 0.0
        dts = np.diff(ts)
        return float(1.0 / np.mean(dts)) if dts.size > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# HDF5 saving
# ═══════════════════════════════════════════════════════════════════════════════


def save_episode_hdf5(episode: dict, path: str, prompt: str, task: str, hz: float) -> None:
    n_steps = len(episode["state"])
    assert n_steps > 0, "Cannot save an empty episode"

    cam1 = np.stack(episode["cam1_rgb"])
    cam2 = np.stack(episode["cam2_rgb"])
    cam3 = np.stack(episode["cam3_rgb"]) if "cam3_rgb" in episode else None
    qpos = np.round(np.stack(episode["state"]), 2)
    action = np.round(np.stack(episode["action"]), 2)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    t0 = time.time()
    with h5py.File(path, "w", rdcc_nbytes=1024**2 * 32) as root:
        root.attrs["sim"] = False
        root.attrs["prompt"] = prompt
        root.attrs["task"] = task
        root.attrs["hz"] = hz
        root.attrs["n_steps"] = n_steps
        root.attrs["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        root.attrs["num_cameras"] = 3 if cam3 is not None else 2

        chunk = (1, IMAGE_H, IMAGE_W, 3)
        obs = root.create_group("observations")
        imgs = obs.create_group("images")
        imgs.create_dataset("exterior_image_1_left", data=cam1, dtype="uint8", chunks=chunk, compression="lzf")
        imgs.create_dataset("wrist_image_left", data=cam2, dtype="uint8", chunks=chunk, compression="lzf")
        if cam3 is not None:
            imgs.create_dataset("front_image_1", data=cam3, dtype="uint8", chunks=chunk, compression="lzf")
        obs.create_dataset("qpos", data=qpos, dtype="float64", compression="gzip", compression_opts=4)
        root.create_dataset("action", data=action, dtype="float64", compression="gzip", compression_opts=4)

    cams_str = "3 cameras" if cam3 is not None else "2 cameras"
    print(f"[Recorder] Saved {n_steps} steps ({cams_str}) → {path}  ({time.time() - t0:.2f} s)")


# ═══════════════════════════════════════════════════════════════════════════════
# Episode index helpers
# ═══════════════════════════════════════════════════════════════════════════════


def get_next_episode_idx(dataset_dir: str) -> int:
    os.makedirs(dataset_dir, exist_ok=True)
    indices = []
    for fname in os.listdir(dataset_dir):
        if fname.startswith("episode_") and fname.endswith(".hdf5"):
            with contextlib.suppress(ValueError):
                indices.append(int(fname[len("episode_") : -len(".hdf5")]))
    return max(indices, default=-1) + 1


def delete_last_episode(dataset_dir: str, episode_idx: int) -> int:
    target_idx = episode_idx - 1
    if target_idx < 0:
        print("[Recorder] No previous episode to delete.")
        return episode_idx
    path = os.path.join(dataset_dir, f"episode_{target_idx}.hdf5")
    if os.path.isfile(path):
        os.remove(path)
        print(f"[Recorder] Deleted {path}")
        return target_idx
    print(f"[Recorder] File not found: {path}")
    return episode_idx


# ═══════════════════════════════════════════════════════════════════════════════
# Preview rendering
# ═══════════════════════════════════════════════════════════════════════════════


# ── UI palette (BGR) ─────────────────────────────────────────────────────────
_F = cv2.FONT_HERSHEY_DUPLEX
_FSM = 0.50  # small
_FMD = 0.65  # medium
_FLG = 0.80  # large
_TH = 1
_STR = 2

_BG_HEADER = (42, 42, 42)
_BG_STATUS = (22, 22, 22)
_BG_CAM = (18, 18, 18)
_DIV = (58, 58, 58)
_WHITE = (230, 230, 230)
_GRAY = (150, 150, 150)
_HINT = (85, 85, 85)
_ACCENT_W = (80, 190, 100)  # green  — WAITING
_ACCENT_R = (55, 55, 210)  # red    — RECORDING
_PROMPT_COL = (80, 185, 220)  # warm cyan


def _txt(img, text, x, y, fs, color, th=_TH):
    cv2.putText(img, text, (x, y), _F, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), _F, fs, color, th, cv2.LINE_AA)


_CAM_LABELS = ["EXTERIOR  CAM 1", "WRIST  CAM 2", "FRONT  CAM 3"]


def build_preview(cam_frames: list, rec_state, n_steps, avg_hz, prompt, episode_idx) -> np.ndarray:
    """Build the preview image.  cam_frames is [cam1_bgr, cam2_bgr] or [cam1_bgr, cam2_bgr, cam3_bgr]."""
    W, H = PREVIEW_W, PREVIEW_H
    N = len(cam_frames)  # 2 or 3
    TW = W * N

    recording = rec_state == RECORDING
    accent = _ACCENT_R if recording else _ACCENT_W
    blink = int(time.time() * 2) % 2 == 0
    prompt_s = (prompt[:52] + "...") if len(prompt) > 52 else (prompt or "—  press  P  to set")

    # ── 1. Camera frames (clean, no text) ────────────────────────────────────
    def prep(img):
        if img is None:
            f = np.full((H, W, 3), _BG_CAM[0], dtype=np.uint8)
            _txt(f, "NO SIGNAL", W // 2 - 62, H // 2 + 6, _FMD, _HINT)
            return f
        f = cv2.resize(img, (W, H))
        if recording:
            cv2.rectangle(f, (0, 0), (W - 1, H - 1), _ACCENT_R, 2)
        # 224x224 center-crop guide -- shows what a 224-px crop of the stored image looks like
        _CROP_SIZE = 224
        cx, cy = W // 2, H // 2
        half = _CROP_SIZE // 2
        x0, y0, x1, y1 = cx - half, cy - half, cx + half - 1, cy + half - 1
        cv2.rectangle(f, (x0, y0), (x1, y1), (0, 0, 220), 1)
        cv2.putText(f, "224", (x0 + 3, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 220), 1, cv2.LINE_AA)
        return f

    prepped = [prep(f) for f in cam_frames]
    cameras = np.concatenate(prepped, axis=1)
    for i in range(1, N):
        cv2.line(cameras, (W * i, 0), (W * i, H), _DIV, 1)

    # ── 2. Header strip (camera labels) ──────────────────────────────────────
    hdr = np.full((HEADER_H, TW, 3), _BG_HEADER[0], dtype=np.uint8)
    cv2.line(hdr, (0, HEADER_H - 1), (TW, HEADER_H - 1), _DIV, 1)
    for i in range(1, N):
        cv2.line(hdr, (W * i, 0), (W * i, HEADER_H), _DIV, 1)
    for i, label in enumerate(_CAM_LABELS[:N]):
        _txt(hdr, label, W * i + 12, 19, _FSM, _GRAY)

    # ── 3. Status panel ───────────────────────────────────────────────────────
    pan = np.full((STATUS_H, TW, 3), _BG_STATUS[0], dtype=np.uint8)
    cv2.line(pan, (0, 0), (TW, 0), _DIV, 1)
    cv2.rectangle(pan, (0, 0), (4, STATUS_H), accent, -1)

    if recording:
        dot_x, dot_y = 22, 26
        if blink:
            cv2.circle(pan, (dot_x, dot_y), 8, _ACCENT_R, -1)
            cv2.circle(pan, (dot_x, dot_y), 8, (255, 255, 255), 1)
        _txt(pan, "REC", dot_x + 15, dot_y + 5, _FMD, _ACCENT_R, _STR)
        _txt(pan, f"Ep {episode_idx}    {n_steps} steps    {avg_hz:.1f} Hz", dot_x + 65, dot_y + 5, _FMD, _WHITE)
        _txt(pan, f"Prompt:  {prompt_s}", 12, 56, _FSM, _PROMPT_COL)
        _txt(pan, "S : stop & save", 12, 78, _FSM, _HINT)
    else:
        _txt(pan, "WAITING", 12, 28, _FLG, _ACCENT_W, _STR)
        _txt(pan, f"Ep {episode_idx}    {n_steps} steps", 130, 28, _FMD, _WHITE)
        p_col = _PROMPT_COL if prompt else _HINT
        _txt(pan, f"Prompt:  {prompt_s}", 12, 56, _FSM, p_col)
        _txt(pan, "P: prompt    B: begin recording    D: delete last    Q: quit", 12, 78, _FSM, _HINT)

    return np.vstack([hdr, cameras, pan])


# ═══════════════════════════════════════════════════════════════════════════════
# Timing diagnostics
# ═══════════════════════════════════════════════════════════════════════════════


def print_dt_diagnosis(step_timestamps: list[float], target_hz: float) -> None:
    if len(step_timestamps) < 2:
        return
    dts = np.diff(step_timestamps)
    mean_dt = float(np.mean(dts))
    std_dt = float(np.std(dts))
    actual_hz = 1.0 / mean_dt if mean_dt > 0 else 0.0
    max_jitter = float(np.max(np.abs(dts - mean_dt))) * 1000
    print(
        f"[Recorder] Timing — target {target_hz:.0f} Hz | "
        f"actual {actual_hz:.2f} Hz | "
        f"step dt {mean_dt * 1000:.1f}±{std_dt * 1000:.1f} ms | "
        f"max jitter {max_jitter:.1f} ms"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main recording loop
# ═══════════════════════════════════════════════════════════════════════════════

WAITING = "WAITING"
RECORDING = "RECORDING"


def run(args) -> None:
    cfg = TASK_CONFIGS.get(args.task, TASK_CONFIGS["default"])
    dataset_dir = args.data_dir or cfg["dataset_dir"]
    max_steps = args.max_steps or cfg["episode_len"]
    hz = args.hz or cfg["hz"]
    dt = 1.0 / hz

    episode_idx = args.episode_idx if args.episode_idx is not None else get_next_episode_idx(dataset_dir)

    # ── startup banner ────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("🤖 OPENPI DATA COLLECTION AUTOMATION")
    print("=" * 55)
    print(f"   Task        : {args.task}")
    print(f"   Dataset dir : {dataset_dir}")
    print(f"   Episode     : {episode_idx}  (auto-detected)")
    print(f"   Max steps   : {max_steps}  @  {hz} Hz")
    print()
    print("Make sure the OpenCV window is selected to use keyboard shortcuts:")
    print("  [P] - Set Task Prompt  (Do this first!)")
    print("  [B] - Begin Recording")
    print("  [S] - Stop & Save Recording")
    print("  [D] - Delete Last Episode")
    print("  [Q] - Quit")
    print("=" * 55)
    print()

    # ── ROS 2 node + spin thread ──────────────────────────────────────────────
    rclpy.init()
    node = Node("pi05_episode_recorder")

    # Start the spinner BEFORE constructing DataBuffer.
    # DataBuffer.__init__ blocks waiting for all four topics; if the spinner
    # hasn't started yet, no callbacks can fire and it will always time out.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    buf = DataBuffer(node)
    rec_pub = node.create_publisher(Bool, "/is_recording", 1)

    def _pub_rec(state: bool):
        rec_pub.publish(Bool(data=state))

    # ── OpenCV preview window (adaptive width) ────────────────────────────────
    num_cams = 3 if buf.has_front_camera else 2
    win_label = "exterior | wrist | front" if buf.has_front_camera else "exterior | wrist"
    win = f"Recorder  [{win_label}]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, PREVIEW_W * num_cams, HEADER_H + PREVIEW_H + STATUS_H)
    cv2.moveWindow(win, 30, 30)

    # ── state + buffers ───────────────────────────────────────────────────────
    rec_state: str = WAITING
    episode_buf: dict = {}
    step_ts: list[float] = []
    prompt: str = ""
    last_step_t: float = 0.0
    last_preview_t: float = 0.0
    PREVIEW_HZ: float = 15.0  # preview renders at 15 Hz; recording runs at full hz

    def reset_buf():
        nonlocal episode_buf, step_ts, last_step_t
        episode_buf = {"cam1_rgb": [], "cam2_rgb": [], "state": [], "action": []}
        if buf.has_front_camera:
            episode_buf["cam3_rgb"] = []
        step_ts = []
        last_step_t = 0.0

    reset_buf()

    # ── main loop ─────────────────────────────────────────────────────────────
    try:
        while rclpy.ok():
            # Recording step runs FIRST — before preview render — so that the
            # ~30-50 ms render block does not delay the next recording tick.
            # waitKey(1) keeps the loop tight so recording hits the full target Hz.
            now_loop = time.time()
            cv_key = cv2.waitKey(1) & 0xFF

            # ── recording step (rate-limited to hz) — highest priority ────────
            if rec_state == RECORDING:
                now = time.time()
                if now - last_step_t >= dt:
                    snap = buf.get_snapshot()
                    if snap is not None:
                        episode_buf["cam1_rgb"].append(preprocess_image(snap["cam1_rgb"]))
                        episode_buf["cam2_rgb"].append(preprocess_image(snap["cam2_rgb"]))
                        if buf.has_front_camera:
                            episode_buf["cam3_rgb"].append(preprocess_image(snap["cam3_rgb"]))
                        episode_buf["state"].append(snap["state"])
                        episode_buf["action"].append(snap["action"])
                        step_ts.append(now)
                        last_step_t += dt  # accumulate to avoid drift

                        if len(step_ts) >= max_steps:
                            rec_state = WAITING
                            path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
                            save_episode_hdf5(episode_buf, path, prompt, args.task, hz)
                            print_dt_diagnosis(step_ts, hz)
                            node.get_logger().info(
                                f"[Recorder] Max steps reached — episode {episode_idx} auto-saved."
                            )
                            episode_idx += 1
                            reset_buf()
                            buf.reset_hz_counter()

            # Preview renders at PREVIEW_HZ (15 Hz) — lower priority than recording.
            if now_loop - last_preview_t >= 1.0 / PREVIEW_HZ:
                cam_frames = buf.get_preview_frames()
                frame = build_preview(cam_frames, rec_state, len(step_ts), buf.avg_hz(), prompt, episode_idx)
                cv2.imshow(win, frame)
                last_preview_t = now_loop

            # Merge cv2 key (OpenCV window focus) with pynput global key.
            key_char: str | None = None
            if cv_key != 255:
                key_char = chr(cv_key)
            with contextlib.suppress(queue.Empty):
                key_char = _key_queue.get_nowait()

            key = ord(key_char) if key_char else 255

            # ── P — Set Task Prompt (any state, in-window text entry) ────────
            if key == ord("p") or key == ord("P"):
                typing = ""
                # Drain any queued keys before starting entry
                while not _key_queue.empty():
                    try:
                        _key_queue.get_nowait()
                    except queue.Empty:
                        break
                while True:
                    cam_frames = buf.get_preview_frames()
                    # Build a full-layout frame with the prompt panel below
                    _frame = build_preview(cam_frames, WAITING, len(step_ts), buf.avg_hz(), prompt, episode_idx)
                    # Overwrite the status panel with the typing prompt
                    pan_y = HEADER_H + PREVIEW_H
                    _frame[pan_y:, :] = _BG_STATUS[0]
                    cv2.line(_frame, (0, pan_y), (PREVIEW_W * num_cams, pan_y), _DIV, 1)
                    cv2.rectangle(_frame, (0, pan_y), (4, pan_y + STATUS_H), _PROMPT_COL, -1)
                    _txt(_frame, "SET PROMPT", 12, pan_y + 26, _FLG, _PROMPT_COL, _STR)
                    _txt(_frame, f"> {typing}_", 12, pan_y + 56, _FMD, _WHITE)
                    _txt(_frame, "Enter: confirm    Esc: cancel    Backspace: delete", 12, pan_y + 78, _FSM, _HINT)
                    cv2.imshow(win, _frame)
                    cv_k = cv2.waitKey(50) & 0xFF
                    # Accept from pynput queue first, then cv2
                    pk: str | None = None
                    with contextlib.suppress(queue.Empty):
                        pk = _key_queue.get_nowait()
                    if pk is not None:
                        if pk in {"\r", "\n"}:
                            break
                        if pk == "\x1b":
                            typing = ""
                            break
                        if pk in ("\x08", "\x7f"):
                            typing = typing[:-1]
                        elif len(pk) == 1 and 32 <= ord(pk) <= 126:
                            typing += pk
                    elif cv_k == 13:  # Enter — confirm
                        break
                    elif cv_k == 27:  # Esc — cancel
                        typing = ""
                        break
                    elif cv_k in (8, 127):  # Backspace
                        typing = typing[:-1]
                    elif 32 <= cv_k <= 126:
                        typing += chr(cv_k)
                if typing:
                    prompt = typing
                    node.get_logger().info(f'[Recorder] Prompt set: "{prompt}"')
                else:
                    node.get_logger().warning("[Recorder] Prompt entry cancelled — prompt unchanged.")

            # ── B — Begin Recording (WAITING only) ────────────────────────────
            elif (key == ord("b") or key == ord("B")) and rec_state == WAITING:
                if not prompt:
                    node.get_logger().warning("[Recorder] No prompt set — press P first!")
                    print("  ⚠️  Press [P] to set a task prompt before recording.")
                else:
                    reset_buf()
                    buf.reset_hz_counter()
                    last_step_t = time.time()
                    rec_state = RECORDING
                    _pub_rec(True)
                    node.get_logger().info(f"[Recorder] *** BEGIN episode {episode_idx} ***")
                    node.get_logger().info(f'[Recorder] Prompt: "{prompt}"')

            # ── S — Stop & Save (RECORDING only) ─────────────────────────────
            elif (key == ord("s") or key == ord("S")) and rec_state == RECORDING:
                rec_state = WAITING
                _pub_rec(False)
                n = len(step_ts)
                if n == 0:
                    node.get_logger().warning("[Recorder] No data recorded — nothing saved.")
                else:
                    path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
                    save_episode_hdf5(episode_buf, path, prompt, args.task, hz)
                    print_dt_diagnosis(step_ts, hz)
                    node.get_logger().info(f"[Recorder] Episode {episode_idx} saved ({n} steps). Ready for next.")
                    episode_idx += 1
                reset_buf()

            # ── D — Delete Last Episode (WAITING only) ────────────────────────
            elif (key == ord("d") or key == ord("D")) and rec_state == WAITING:
                episode_idx = delete_last_episode(dataset_dir, episode_idx)

            # ── Q — Quit (any state) ──────────────────────────────────────────
            elif key == ord("q") or key == ord("Q"):
                if rec_state == RECORDING:
                    node.get_logger().warning("[Recorder] Quit while recording — data NOT saved.")
                node.get_logger().info("[Recorder] Quitting — shutting down all pipeline nodes.")
                with contextlib.suppress(Exception):
                    os.killpg(os.getpgrp(), signal.SIGTERM)
                break

    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="pi0.5 episode recorder for Uarm → UR5 teleoperation (ROS 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task", default="default", help=f"Task key in TASK_CONFIGS: {list(TASK_CONFIGS)}")
    p.add_argument("--data-dir", default=None, help="Override dataset directory from TASK_CONFIGS")
    p.add_argument("--episode-idx", type=int, default=None, help="Starting episode index (auto-detected if omitted)")
    p.add_argument("--max-steps", type=int, default=None, help="Max timesteps per episode (overrides TASK_CONFIGS)")
    p.add_argument("--hz", type=float, default=None, help="Recording frequency in Hz (overrides TASK_CONFIGS)")
    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()

    def _sigint(_sig, _frame):
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    try:
        run(args)
    except Exception as e:
        print(f"[Recorder] Fatal error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
