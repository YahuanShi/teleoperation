#!/usr/bin/env python3
"""
Episode Recorder
======================

Subscribes to Uarm / UR5 teleoperation ROS topics and records demonstration
episodes in HDF5 format directly compatible with openpi / LeRobot training.

Topics consumed:
    /cam_1         sensor_msgs/Image       → exterior camera
    /cam_2         sensor_msgs/Image       → wrist / gripper camera
    /robot_state   Float64MultiArray (7,)  → actual joints [deg x6] + gripper [0-1]
    /robot_action  Float64MultiArray (7,)  → commanded action (same layout)

HDF5 layout written per episode (aloha-compatible, pi0.5 ready):
    /observations/images/exterior_image_1_left  (T, 224, 224, 3) uint8  RGB
    /observations/images/wrist_image_left        (T, 224, 224, 3) uint8  RGB
    /observations/qpos                           (T, 7) float64
    /action                                      (T, 7) float64
    root.attrs: sim, prompt, task, hz, timestamp, n_steps

Keyboard shortcuts (focus the OpenCV preview window):
    P  →  Set task prompt      type once, reused for every episode in this session
    B  →  Begin recording      requires prompt to be set first
    S  →  Stop & Save          stops recording and immediately writes HDF5 to disk
    D  →  Delete last episode  removes the previously saved HDF5 file (undo save)
    Q  →  Quit                 safely exits (Ctrl-C also works)

Usage:
    python3 episode_recorder.py
    python3 episode_recorder.py --task pick_place
    python3 episode_recorder.py --task pick_place --data-dir ~/my_data --hz 20
    python3 episode_recorder.py --task pick_place --episode-idx 5
"""

import argparse
from collections import deque
import contextlib
import datetime
import os
import signal
import sys
import threading
import time

import queue

import cv2
from cv_bridge import CvBridge
import h5py
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64MultiArray

# Global keyboard listener (works regardless of which window has focus).
# Falls back gracefully if pynput is unavailable.
_key_queue: queue.SimpleQueue = queue.SimpleQueue()
try:
    from pynput import keyboard as _kb

    def _on_press(key):
        try:
            _key_queue.put(key.char)
        except AttributeError:
            pass  # special key (shift, ctrl, …) — ignore

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
        "episode_len": 1000,
        "hz": 10,
    },
    # "pick_place": {
    #     "dataset_dir": os.path.join(_OPENPI_ROOT, "dataset", f"ur5_pick_place_{_TODAY}"),
    #     "episode_len": 300,
    #     "hz": 10,
    # },
}

IMAGE_H, IMAGE_W = 224, 224
PREVIEW_H, PREVIEW_W = 480, 640   # per-camera frame size
HEADER_H  = 28                    # camera label strip above images
STATUS_H  = 88                    # info panel below images

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
    """

    def __init__(self, node: Node):
        self._node = node
        self._lock = threading.Lock()
        self._bridge = CvBridge()

        self._cam1_bgr: np.ndarray | None = None
        self._cam2_bgr: np.ndarray | None = None
        self._state: np.ndarray | None = None
        self._action: np.ndarray | None = None

        self._step_ts: deque = deque(maxlen=200)
        self._last_warn: dict[str, float] = {}

        node.create_subscription(Image, "/cam_1", self._cb_cam1, 10)
        node.create_subscription(Image, "/cam_2", self._cb_cam2, 10)
        node.create_subscription(Float64MultiArray, "/robot_state", self._cb_state, 10)
        node.create_subscription(Float64MultiArray, "/robot_action", self._cb_action, 10)

        node.get_logger().info("[Recorder] Waiting for all four topics …")
        deadline = time.time() + 30.0
        while not self.is_ready():
            if not rclpy.ok():
                raise RuntimeError("ROS shutdown while waiting for topics")
            if time.time() > deadline:
                raise RuntimeError("Topics not received after 30 s — ensure the teleoperation nodes are running.")
            time.sleep(0.1)
        node.get_logger().info("[Recorder] All topics ready.")

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

    def _cb_state(self, msg: Float64MultiArray):
        with self._lock:
            self._state = np.array(msg.data, dtype=np.float64)

    def _cb_action(self, msg: Float64MultiArray):
        with self._lock:
            self._action = np.array(msg.data, dtype=np.float64)

    # ── public API ────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        with self._lock:
            return all(x is not None for x in [self._cam1_bgr, self._cam2_bgr, self._state, self._action])

    def get_snapshot(self) -> dict | None:
        with self._lock:
            # Inline check — do NOT call is_ready() here (it also acquires _lock → deadlock)
            if any(x is None for x in [self._cam1_bgr, self._cam2_bgr, self._state, self._action]):
                return None
            snap = {
                "cam1_rgb": cv2.cvtColor(self._cam1_bgr, cv2.COLOR_BGR2RGB),
                "cam2_rgb": cv2.cvtColor(self._cam2_bgr, cv2.COLOR_BGR2RGB),
                "state": self._state.copy(),
                "action": self._action.copy(),
            }
        self._step_ts.append(time.time())
        return snap

    def get_preview_frames(self) -> tuple:
        with self._lock:
            c1 = self._cam1_bgr.copy() if self._cam1_bgr is not None else None
            c2 = self._cam2_bgr.copy() if self._cam2_bgr is not None else None
        return c1, c2

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
    qpos = np.stack(episode["state"])
    action = np.stack(episode["action"])

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    t0 = time.time()
    with h5py.File(path, "w", rdcc_nbytes=1024**2 * 4) as root:
        root.attrs["sim"] = False
        root.attrs["prompt"] = prompt
        root.attrs["task"] = task
        root.attrs["hz"] = hz
        root.attrs["n_steps"] = n_steps
        root.attrs["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        chunk = (1, IMAGE_H, IMAGE_W, 3)
        obs = root.create_group("observations")
        imgs = obs.create_group("images")
        imgs.create_dataset("exterior_image_1_left", data=cam1, dtype="uint8", chunks=chunk, compression="lzf")
        imgs.create_dataset("wrist_image_left", data=cam2, dtype="uint8", chunks=chunk, compression="lzf")
        obs.create_dataset("qpos", data=qpos, dtype="float64", compression="gzip", compression_opts=4)
        root.create_dataset("action", data=action, dtype="float64", compression="gzip", compression_opts=4)

    print(f"[Recorder] Saved {n_steps} steps → {path}  ({time.time() - t0:.2f} s)")


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
_F   = cv2.FONT_HERSHEY_DUPLEX
_FSM = 0.50   # small
_FMD = 0.65   # medium
_FLG = 0.80   # large
_TH  = 1
_STR = 2

_BG_HEADER  = (42,  42,  42)
_BG_STATUS  = (22,  22,  22)
_BG_CAM     = (18,  18,  18)
_DIV        = (58,  58,  58)
_WHITE      = (230, 230, 230)
_GRAY       = (150, 150, 150)
_HINT       = (85,  85,  85)
_ACCENT_W   = (80,  190, 100)   # green  — WAITING
_ACCENT_R   = (55,  55,  210)   # red    — RECORDING
_PROMPT_COL = (80,  185, 220)   # warm cyan


def _txt(img, text, x, y, fs, color, th=_TH):
    cv2.putText(img, text, (x, y), _F, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), _F, fs, color,    th,     cv2.LINE_AA)


def build_preview(cam1_bgr, cam2_bgr, rec_state, n_steps, avg_hz, prompt, episode_idx) -> np.ndarray:
    W, H = PREVIEW_W, PREVIEW_H
    TW   = W * 2   # total width

    recording = rec_state == RECORDING
    accent    = _ACCENT_R if recording else _ACCENT_W
    blink     = int(time.time() * 2) % 2 == 0
    prompt_s  = (prompt[:52] + "...") if len(prompt) > 52 else (prompt or "—  press  P  to set")

    # ── 1. Camera frames (clean, no text) ────────────────────────────────────
    def prep(img):
        if img is None:
            f = np.full((H, W, 3), _BG_CAM[0], dtype=np.uint8)
            _txt(f, "NO SIGNAL", W // 2 - 62, H // 2 + 6, _FMD, _HINT)
            return f
        f = cv2.resize(img, (W, H))
        if recording:
            cv2.rectangle(f, (0, 0), (W - 1, H - 1), _ACCENT_R, 2)
        return f

    cam1 = prep(cam1_bgr)
    cam2 = prep(cam2_bgr)
    cameras = np.concatenate([cam1, cam2], axis=1)
    cv2.line(cameras, (W, 0), (W, H), _DIV, 1)   # center divider

    # ── 2. Header strip (camera labels) ──────────────────────────────────────
    hdr = np.full((HEADER_H, TW, 3), _BG_HEADER[0], dtype=np.uint8)
    cv2.line(hdr, (0, HEADER_H - 1), (TW, HEADER_H - 1), _DIV, 1)
    cv2.line(hdr, (W, 0), (W, HEADER_H), _DIV, 1)
    _txt(hdr, "EXTERIOR  CAM 1", 12, 19, _FSM, _GRAY)
    _txt(hdr, "WRIST  CAM 2",   W + 12, 19, _FSM, _GRAY)

    # ── 3. Status panel ───────────────────────────────────────────────────────
    pan = np.full((STATUS_H, TW, 3), _BG_STATUS[0], dtype=np.uint8)
    cv2.line(pan, (0, 0), (TW, 0), _DIV, 1)          # top border
    cv2.rectangle(pan, (0, 0), (4, STATUS_H), accent, -1)   # accent stripe

    if recording:
        # Row 1: REC badge + episode/steps/hz
        dot_x, dot_y = 22, 26
        if blink:
            cv2.circle(pan, (dot_x, dot_y), 8, _ACCENT_R, -1)
            cv2.circle(pan, (dot_x, dot_y), 8, (255, 255, 255), 1)
        _txt(pan, "REC", dot_x + 15, dot_y + 5, _FMD, _ACCENT_R, _STR)
        _txt(pan, f"Ep {episode_idx}    {n_steps} steps    {avg_hz:.1f} Hz",
             dot_x + 65, dot_y + 5, _FMD, _WHITE)
        # Row 2: prompt
        _txt(pan, f"Prompt:  {prompt_s}", 12, 56, _FSM, _PROMPT_COL)
        # Row 3: hint
        _txt(pan, "S : stop & save", 12, 78, _FSM, _HINT)
    else:
        # Row 1: state badge + episode
        _txt(pan, "WAITING", 12, 28, _FLG, _ACCENT_W, _STR)
        _txt(pan, f"Ep {episode_idx}    {n_steps} steps", 130, 28, _FMD, _WHITE)
        # Row 2: prompt
        p_col = _PROMPT_COL if prompt else _HINT
        _txt(pan, f"Prompt:  {prompt_s}", 12, 56, _FSM, p_col)
        # Row 3: hints
        _txt(pan, "P: prompt    B: begin recording    D: delete last    Q: quit",
             12, 78, _FSM, _HINT)

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

    # ── OpenCV preview window ─────────────────────────────────────────────────
    win = "Recorder  [exterior | wrist]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, PREVIEW_W * 2, HEADER_H + PREVIEW_H + STATUS_H)
    cv2.moveWindow(win, 30, 30)   # top-left of monitor (adjust if needed)

    # ── state + buffers ───────────────────────────────────────────────────────
    rec_state: str = WAITING
    episode_buf: dict = {}
    step_ts: list[float] = []
    prompt: str = ""
    last_step_t: float = 0.0

    def reset_buf():
        nonlocal episode_buf, step_ts, last_step_t
        episode_buf = {"cam1_rgb": [], "cam2_rgb": [], "state": [], "action": []}
        step_ts = []
        last_step_t = 0.0

    reset_buf()

    # ── main loop ─────────────────────────────────────────────────────────────
    try:
        while rclpy.ok():
            c1_bgr, c2_bgr = buf.get_preview_frames()
            frame = build_preview(c1_bgr, c2_bgr, rec_state, len(step_ts), buf.avg_hz(), prompt, episode_idx)
            cv2.imshow(win, frame)
            cv_key = cv2.waitKey(10) & 0xFF

            # Merge cv2 key (OpenCV window focus) with pynput global key.
            key_char: str | None = None
            if cv_key != 255:
                key_char = chr(cv_key)
            try:
                key_char = _key_queue.get_nowait()
            except queue.Empty:
                pass

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
                    c1_bgr, c2_bgr = buf.get_preview_frames()
                    # Build a full-layout frame with the prompt panel below
                    _frame = build_preview(c1_bgr, c2_bgr, WAITING, len(step_ts), buf.avg_hz(), prompt, episode_idx)
                    # Overwrite the status panel with the typing prompt
                    pan_y = HEADER_H + PREVIEW_H
                    _frame[pan_y:, :] = _BG_STATUS[0]
                    cv2.line(_frame, (0, pan_y), (PREVIEW_W * 2, pan_y), _DIV, 1)
                    cv2.rectangle(_frame, (0, pan_y), (4, pan_y + STATUS_H), _PROMPT_COL, -1)
                    _txt(_frame, "SET PROMPT", 12, pan_y + 26, _FLG, _PROMPT_COL, _STR)
                    _txt(_frame, f"> {typing}_", 12, pan_y + 56, _FMD, _WHITE)
                    _txt(_frame, "Enter: confirm    Esc: cancel    Backspace: delete", 12, pan_y + 78, _FSM, _HINT)
                    cv2.imshow(win, _frame)
                    cv_k = cv2.waitKey(50) & 0xFF
                    # Accept from pynput queue first, then cv2
                    pk: str | None = None
                    try:
                        pk = _key_queue.get_nowait()
                    except queue.Empty:
                        pass
                    if pk is not None:
                        if pk == "\r" or pk == "\n":
                            break
                        elif pk == "\x1b":
                            typing = ""
                            break
                        elif pk in ("\x08", "\x7f"):
                            typing = typing[:-1]
                        elif len(pk) == 1 and 32 <= ord(pk) <= 126:
                            typing += pk
                    elif cv_k == 13:   # Enter — confirm
                        break
                    elif cv_k == 27:   # Esc — cancel
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
                try:
                    os.killpg(os.getpgrp(), signal.SIGTERM)
                except Exception:
                    pass
                break

            # ── recording step (rate-limited to hz) ───────────────────────────
            if rec_state == RECORDING:
                now = time.time()
                if now - last_step_t >= dt:
                    snap = buf.get_snapshot()
                    if snap is not None:
                        episode_buf["cam1_rgb"].append(preprocess_image(snap["cam1_rgb"]))
                        episode_buf["cam2_rgb"].append(preprocess_image(snap["cam2_rgb"]))
                        episode_buf["state"].append(snap["state"])
                        episode_buf["action"].append(snap["action"])
                        step_ts.append(now)
                        last_step_t = now

                        if len(step_ts) >= max_steps:
                            rec_state = WAITING
                            path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
                            node.get_logger().info(f"[Recorder] Max steps ({max_steps}) reached — auto-saving …")
                            save_episode_hdf5(episode_buf, path, prompt, args.task, hz)
                            print_dt_diagnosis(step_ts, hz)
                            episode_idx += 1
                            reset_buf()

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
