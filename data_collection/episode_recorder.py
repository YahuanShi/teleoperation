#!/usr/bin/env python3
"""
pi0.5 Episode Recorder
======================

Subscribes to Uarm / UR5 teleoperation ROS topics and records demonstration
episodes in HDF5 format directly compatible with openpi / LeRobot training.

Topics consumed:
    /cam_1         sensor_msgs/Image       → exterior camera
    /cam_2         sensor_msgs/Image       → wrist / gripper camera
    /robot_state   Float64MultiArray (7,)  → actual joints [deg ×6] + gripper [0-1]
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
import os
import signal
import sys
import time
import threading
from collections import deque

import cv2
import h5py
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Add / edit tasks here.  CLI flag --task selects which entry to use.
TASK_CONFIGS: dict = {
    "default": {
        "dataset_dir": os.path.expanduser("~/pi05_dataset/default"),
        "episode_len": 500,   # max timesteps before auto-stop + auto-save
        "hz": 15,             # recording frequency
    },
    # Example additional task:
    # "pick_place": {
    #     "dataset_dir": os.path.expanduser("~/pi05_dataset/pick_place"),
    #     "episode_len": 300,
    #     "hz": 15,
    # },
}

IMAGE_H, IMAGE_W   = 224, 224      # pi0.5 standard image resolution
PREVIEW_H, PREVIEW_W = 360, 480   # per-camera panel size in the preview window

# ═══════════════════════════════════════════════════════════════════════════════
# Image preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def center_crop_resize(image: np.ndarray,
                        out_h: int = IMAGE_H,
                        out_w: int = IMAGE_W) -> np.ndarray:
    """
    Square-crop the centre of the image then resize to (out_h, out_w).
    Preferred for robot cameras where the scene is centred.
    Input/output: RGB uint8.
    """
    h, w    = image.shape[:2]
    side    = min(h, w)
    y0, x0  = (h - side) // 2, (w - side) // 2
    cropped = image[y0:y0 + side, x0:x0 + side]
    return cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)


def resize_with_pad(image: np.ndarray,
                    out_h: int = IMAGE_H,
                    out_w: int = IMAGE_W) -> np.ndarray:
    """
    Resize preserving aspect ratio, pad remainder with black.
    Use when full field-of-view matters more than a square crop.
    """
    h, w    = image.shape[:2]
    scale   = min(out_h / h, out_w / w)
    nh, nw  = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas  = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    y0, x0  = (out_h - nh) // 2, (out_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


# Active preprocessing — swap to resize_with_pad if preferred.
preprocess_image = center_crop_resize

# ═══════════════════════════════════════════════════════════════════════════════
# Thread-safe ROS data buffer
# ═══════════════════════════════════════════════════════════════════════════════

class DataBuffer:
    """
    Subscribes to all four ROS topics and keeps the most recent sample of each.
    Thread-safe: ROS callbacks run in background threads; main thread reads here.
    """

    def __init__(self):
        self._lock   = threading.Lock()
        self._bridge = CvBridge()

        self._cam1_bgr: np.ndarray | None = None   # /cam_1  exterior
        self._cam2_bgr: np.ndarray | None = None   # /cam_2  wrist
        self._state:    np.ndarray | None = None   # /robot_state  (7,)
        self._action:   np.ndarray | None = None   # /robot_action (7,)

        # Rolling timestamps for Hz diagnostics
        self._step_ts: deque = deque(maxlen=200)

        rospy.Subscriber("/cam_1",        Image,             self._cb_cam1)
        rospy.Subscriber("/cam_2",        Image,             self._cb_cam2)
        rospy.Subscriber("/robot_state",  Float64MultiArray, self._cb_state)
        rospy.Subscriber("/robot_action", Float64MultiArray, self._cb_action)

        rospy.loginfo("[Recorder] Waiting for all four topics …")
        deadline = time.time() + 30.0
        while not self.is_ready():
            if rospy.is_shutdown():
                raise RuntimeError("ROS shutdown while waiting for topics")
            if time.time() > deadline:
                raise RuntimeError(
                    "Topics not received after 30 s — ensure the "
                    "teleoperation nodes are running.")
            time.sleep(0.1)
        rospy.loginfo("[Recorder] All topics ready.")

    # ── callbacks ──────────────────────────────────────────────────────────────

    def _cb_cam1(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._lock:
                self._cam1_bgr = frame
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[Recorder] cam_1 error: {e}")

    def _cb_cam2(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._lock:
                self._cam2_bgr = frame
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[Recorder] cam_2 error: {e}")

    def _cb_state(self, msg: Float64MultiArray):
        with self._lock:
            self._state = np.array(msg.data, dtype=np.float64)

    def _cb_action(self, msg: Float64MultiArray):
        with self._lock:
            self._action = np.array(msg.data, dtype=np.float64)

    # ── public API ─────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        with self._lock:
            return all(x is not None for x in
                       [self._cam1_bgr, self._cam2_bgr, self._state, self._action])

    def get_snapshot(self) -> dict | None:
        """Deep-copy snapshot (BGR→RGB). Returns None if any topic missing."""
        with self._lock:
            if not self.is_ready():
                return None
            snap = {
                "cam1_rgb": cv2.cvtColor(self._cam1_bgr, cv2.COLOR_BGR2RGB),
                "cam2_rgb": cv2.cvtColor(self._cam2_bgr, cv2.COLOR_BGR2RGB),
                "state":    self._state.copy(),
                "action":   self._action.copy(),
            }
        self._step_ts.append(time.time())
        return snap

    def get_preview_frames(self) -> tuple:
        """Return raw BGR copies of both cameras for display."""
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

def save_episode_hdf5(episode: dict, path: str, prompt: str,
                       task: str, hz: float) -> None:
    """
    Write one episode to HDF5 in pi0.5 / aloha-compatible format.
    episode keys: cam1_rgb, cam2_rgb, state, action  (lists of np.ndarray)
    """
    T = len(episode["state"])
    assert T > 0, "Cannot save an empty episode"

    cam1   = np.stack(episode["cam1_rgb"])   # (T, H, W, 3)
    cam2   = np.stack(episode["cam2_rgb"])   # (T, H, W, 3)
    qpos   = np.stack(episode["state"])      # (T, 7)
    action = np.stack(episode["action"])     # (T, 7)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    t0 = time.time()
    with h5py.File(path, "w", rdcc_nbytes=1024 ** 2 * 4) as root:
        root.attrs["sim"]       = False
        root.attrs["prompt"]    = prompt
        root.attrs["task"]      = task
        root.attrs["hz"]        = hz
        root.attrs["n_steps"]   = T
        root.attrs["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        chunk = (1, IMAGE_H, IMAGE_W, 3)
        obs   = root.create_group("observations")
        imgs  = obs.create_group("images")
        # lzf: fast compression, good ratio for uint8 image data
        imgs.create_dataset("exterior_image_1_left", data=cam1,
                            dtype="uint8", chunks=chunk, compression="lzf")
        imgs.create_dataset("wrist_image_left", data=cam2,
                            dtype="uint8", chunks=chunk, compression="lzf")

        obs.create_dataset("qpos",    data=qpos,
                           dtype="float64", compression="gzip", compression_opts=4)
        root.create_dataset("action", data=action,
                            dtype="float64", compression="gzip", compression_opts=4)

    rospy.loginfo(f"[Recorder] Saved {T} steps → {path}  ({time.time()-t0:.2f} s)")

# ═══════════════════════════════════════════════════════════════════════════════
# Episode index helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_next_episode_idx(dataset_dir: str) -> int:
    os.makedirs(dataset_dir, exist_ok=True)
    indices = []
    for fname in os.listdir(dataset_dir):
        if fname.startswith("episode_") and fname.endswith(".hdf5"):
            try:
                indices.append(int(fname[len("episode_"):-len(".hdf5")]))
            except ValueError:
                pass
    return max(indices, default=-1) + 1


def delete_last_episode(dataset_dir: str, episode_idx: int) -> int:
    """
    Delete episode_{episode_idx - 1}.hdf5 and return the decremented index.
    Does nothing and returns episode_idx unchanged if nothing to delete.
    """
    target_idx = episode_idx - 1
    if target_idx < 0:
        rospy.logwarn("[Recorder] No previous episode to delete.")
        return episode_idx
    path = os.path.join(dataset_dir, f"episode_{target_idx}.hdf5")
    if os.path.isfile(path):
        os.remove(path)
        rospy.loginfo(f"[Recorder] Deleted {path}")
        return target_idx
    else:
        rospy.logwarn(f"[Recorder] File not found: {path}")
        return episode_idx

# ═══════════════════════════════════════════════════════════════════════════════
# Preview rendering
# ═══════════════════════════════════════════════════════════════════════════════

def _put_text(img: np.ndarray, lines: list[str],
               color: tuple = (255, 255, 255)) -> np.ndarray:
    out = img.copy()
    for i, line in enumerate(lines):
        y = 36 + 30 * i
        cv2.putText(out, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,   2, cv2.LINE_AA)
    return out


def build_preview(cam1_bgr: np.ndarray | None,
                  cam2_bgr: np.ndarray | None,
                  rec_state: str,
                  n_steps: int,
                  avg_hz: float,
                  prompt: str,
                  episode_idx: int) -> np.ndarray:
    """Side-by-side [exterior | wrist] with per-state status overlay."""

    def prep(img):
        if img is None:
            return np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)
        return cv2.resize(img, (PREVIEW_W, PREVIEW_H))

    left  = prep(cam1_bgr)
    right = prep(cam2_bgr)

    prompt_disp = (f'"{prompt[:32]}\u2026"' if len(prompt) > 32
                   else (f'"{prompt}"' if prompt else "[ No prompt — press P ]"))
    blink = int(time.time() * 2) % 2 == 0

    if rec_state == "WAITING":
        prompt_color = (0, 220, 220) if prompt else (0, 100, 220)
        left = _put_text(left,
                         [f"Episode {episode_idx}  |  WAITING",
                          prompt_disp,
                          "P: set prompt    B: begin",
                          "D: delete last   Q: quit"],
                         color=(200, 200, 200))
        right = _put_text(right, [prompt_disp], color=prompt_color)

    elif rec_state == "RECORDING":
        dot  = "● REC" if blink else "  REC"
        left = _put_text(left,
                         [f"Episode {episode_idx}  |  {dot}",
                          f"{n_steps} steps   {avg_hz:.1f} Hz",
                          prompt_disp,
                          "S: stop & save"],
                         color=(0, 60, 230))
        right = _put_text(right, [prompt_disp], color=(0, 200, 230))
        # Red recording border
        cv2.rectangle(left, (0, 0), (PREVIEW_W - 1, PREVIEW_H - 1),
                      (0, 0, 210), 5)
        cv2.rectangle(right, (0, 0), (PREVIEW_W - 1, PREVIEW_H - 1),
                      (0, 0, 210), 5)

    return np.concatenate([left, right], axis=1)

# ═══════════════════════════════════════════════════════════════════════════════
# Timing diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def print_dt_diagnosis(step_timestamps: list[float], target_hz: float) -> None:
    if len(step_timestamps) < 2:
        return
    dts       = np.diff(step_timestamps)
    mean_dt   = float(np.mean(dts))
    std_dt    = float(np.std(dts))
    actual_hz = 1.0 / mean_dt if mean_dt > 0 else 0.0
    max_jitter = float(np.max(np.abs(dts - mean_dt))) * 1000
    rospy.loginfo(
        f"[Recorder] Timing — target {target_hz:.0f} Hz | "
        f"actual {actual_hz:.2f} Hz | "
        f"step dt {mean_dt*1000:.1f}±{std_dt*1000:.1f} ms | "
        f"max jitter {max_jitter:.1f} ms"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Main recording loop
# ═══════════════════════════════════════════════════════════════════════════════

WAITING   = "WAITING"
RECORDING = "RECORDING"


def run(args) -> None:
    cfg         = TASK_CONFIGS.get(args.task, TASK_CONFIGS["default"])
    dataset_dir = args.data_dir  or cfg["dataset_dir"]
    max_steps   = args.max_steps or cfg["episode_len"]
    hz          = args.hz        or cfg["hz"]
    dt          = 1.0 / hz

    episode_idx = (args.episode_idx if args.episode_idx is not None
                   else get_next_episode_idx(dataset_dir))

    # ── startup banner ─────────────────────────────────────────────────────────
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

    buf = DataBuffer()

    # ── OpenCV preview window ──────────────────────────────────────────────────
    WIN = "🤖 pi0.5 Recorder  [exterior | wrist]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, PREVIEW_W * 2, PREVIEW_H)

    # ── state + buffers ────────────────────────────────────────────────────────
    rec_state   : str         = WAITING
    episode_buf : dict        = {}
    step_ts     : list[float] = []
    prompt      : str         = ""        # set once with P, reused for all episodes
    last_step_t : float       = 0.0

    def reset_buf():
        nonlocal episode_buf, step_ts, last_step_t
        episode_buf = {"cam1_rgb": [], "cam2_rgb": [], "state": [], "action": []}
        step_ts     = []
        last_step_t = 0.0

    reset_buf()

    # ── main loop ──────────────────────────────────────────────────────────────
    while not rospy.is_shutdown():

        # display
        c1_bgr, c2_bgr = buf.get_preview_frames()
        frame = build_preview(c1_bgr, c2_bgr, rec_state,
                               len(step_ts), buf.avg_hz(), prompt, episode_idx)
        cv2.imshow(WIN, frame)
        key = cv2.waitKey(10) & 0xFF      # ~100 Hz display refresh

        # ── P — Set Task Prompt (any state) ───────────────────────────────────
        if key == ord('p') or key == ord('P'):
            cv2.destroyAllWindows()
            print()
            new_prompt = input("  [P] Enter task prompt for this session: ").strip()
            if new_prompt:
                prompt = new_prompt
                rospy.loginfo(f"[Recorder] Prompt set: \"{prompt}\"")
            else:
                rospy.logwarn("[Recorder] Empty input — prompt unchanged.")
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN, PREVIEW_W * 2, PREVIEW_H)

        # ── B — Begin Recording (WAITING only) ────────────────────────────────
        elif (key == ord('b') or key == ord('B')) and rec_state == WAITING:
            if not prompt:
                rospy.logwarn("[Recorder] No prompt set — press P first!")
                print("  ⚠️  Press [P] to set a task prompt before recording.")
            else:
                reset_buf()
                last_step_t = time.time()
                rec_state   = RECORDING
                rospy.loginfo(f"[Recorder] *** BEGIN episode {episode_idx} ***")
                rospy.loginfo(f"[Recorder] Prompt: \"{prompt}\"")

        # ── S — Stop & Save (RECORDING only) ──────────────────────────────────
        elif (key == ord('s') or key == ord('S')) and rec_state == RECORDING:
            rec_state = WAITING
            n = len(step_ts)
            if n == 0:
                rospy.logwarn("[Recorder] No data recorded — nothing saved.")
            else:
                path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
                save_episode_hdf5(episode_buf, path, prompt, args.task, hz)
                print_dt_diagnosis(step_ts, hz)
                rospy.loginfo(f"[Recorder] Episode {episode_idx} saved "
                              f"({n} steps). Ready for next.")
                episode_idx += 1
            reset_buf()

        # ── D — Delete Last Episode (WAITING only) ────────────────────────────
        elif (key == ord('d') or key == ord('D')) and rec_state == WAITING:
            episode_idx = delete_last_episode(dataset_dir, episode_idx)

        # ── Q — Quit (any state) ──────────────────────────────────────────────
        elif key == ord('q') or key == ord('Q'):
            if rec_state == RECORDING:
                rospy.logwarn("[Recorder] Quit while recording — data NOT saved.")
            rospy.loginfo("[Recorder] Quitting.")
            break

        # ── recording step (rate-limited to hz) ───────────────────────────────
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
                        # Auto stop & save when max steps reached
                        rec_state = WAITING
                        path = os.path.join(dataset_dir,
                                            f"episode_{episode_idx}.hdf5")
                        rospy.loginfo(
                            f"[Recorder] Max steps ({max_steps}) reached — "
                            "auto-saving …")
                        save_episode_hdf5(episode_buf, path, prompt, args.task, hz)
                        print_dt_diagnosis(step_ts, hz)
                        episode_idx += 1
                        reset_buf()

    cv2.destroyAllWindows()

# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="pi0.5 episode recorder for Uarm → UR5 teleoperation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task", default="default",
                   help=f"Task key in TASK_CONFIGS: {list(TASK_CONFIGS)}")
    p.add_argument("--data-dir", default=None,
                   help="Override dataset directory from TASK_CONFIGS")
    p.add_argument("--episode-idx", type=int, default=None,
                   help="Starting episode index (auto-detected if omitted)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Max timesteps per episode (overrides TASK_CONFIGS)")
    p.add_argument("--hz", type=float, default=None,
                   help="Recording frequency in Hz (overrides TASK_CONFIGS)")
    return p


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    rospy.init_node("pi05_episode_recorder", anonymous=False)

    def _sigint(sig, frame):
        rospy.signal_shutdown("Ctrl-C")
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)

    try:
        run(args)
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"[Recorder] Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
