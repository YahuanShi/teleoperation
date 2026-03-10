#!/usr/bin/env python3
"""
Replay a recorded UR5 HDF5 episode - offline viewer, no ROS required.

Usage:
    # Play a specific episode file:
    python3 teleoperation/data_collection/replay_episode.py path/to/episode_0.hdf5

    # Pick episode from a dataset directory (defaults to episode 0):
    python3 teleoperation/data_collection/replay_episode.py path/to/ur5_dataset_20260227/

    # Pick a specific episode index from a directory:
    python3 teleoperation/data_collection/replay_episode.py path/to/ur5_dataset_20260227/ -e 3

Controls (focus the OpenCV window):
    Space        Pause / resume
    -> or .      Step forward  one frame (pauses)
    <- or ,      Step backward one frame (pauses)
    + or =       Speed up  (halve inter-frame delay)
    - or _       Slow down (double inter-frame delay)
    R            Restart from frame 0
    ESC or Q     Quit
"""

import argparse
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np

# ── Display constants ─────────────────────────────────────────────────────────
_SCALE = 2  # scale factor applied to each 224x224 camera image -> 448x448
_INFO_H = 168  # height of the text info panel below cameras
_PAD = 8  # pixel gap between panels

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GREEN = (100, 255, 100)
_YELLOW = (80, 230, 230)
_WHITE = (220, 220, 220)
_GRAY = (140, 140, 140)


# ── Data loading ──────────────────────────────────────────────────────────────


def load_episode(hdf5_path: Path) -> dict:
    with h5py.File(hdf5_path, "r") as f:
        return {
            "qpos": f["/observations/qpos"][:],  # (T,7) deg
            "action": f["/action"][:],  # (T,7) deg
            "ext": f["/observations/images/exterior_image_1_left"][:],  # (T,224,224,3) RGB
            "wrist": f["/observations/images/wrist_image_left"][:],  # (T,224,224,3) RGB
            "prompt": str(f.attrs.get("prompt", "")),
            "hz": float(f.attrs.get("hz", 15)),
        }


def resolve_path(raw: Path, episode_idx: int) -> Path:
    if raw.is_file():
        return raw
    if raw.is_dir():
        candidates = sorted(raw.glob("episode_*.hdf5"))
        if not candidates:
            sys.exit(f"No episode_*.hdf5 files found in {raw}")
        if episode_idx >= len(candidates):
            sys.exit(f"Episode index {episode_idx} out of range (found {len(candidates)} episodes in {raw})")
        return candidates[episode_idx]
    sys.exit(f"Path not found: {raw}")


# ── Rendering ─────────────────────────────────────────────────────────────────


def _put(img, text, pos, scale=0.46, color=_WHITE, thickness=1):
    cv2.putText(img, text, pos, _FONT, scale, color, thickness, cv2.LINE_AA)


def make_info_panel(
    width: int,
    t: int,
    n_total: int,
    qpos: np.ndarray,
    action: np.ndarray,
    prompt: str,
    *,
    paused: bool,
    delay_ms: int,
) -> np.ndarray:
    panel = np.zeros((_INFO_H, width, 3), dtype=np.uint8)

    # ── Progress bar ──────────────────────────────────────────────────────
    bx, by, bh = _PAD, 14, 8
    bw = width - _PAD * 2
    cv2.rectangle(panel, (bx, by - bh // 2), (bx + bw, by + bh // 2), (50, 50, 50), -1)
    filled = int(bw * t / max(n_total - 1, 1))
    cv2.rectangle(panel, (bx, by - bh // 2), (bx + filled, by + bh // 2), (0, 180, 255), -1)

    # ── Status ────────────────────────────────────────────────────────────
    status = "PAUSED" if paused else f"PLAY  {1000 / max(delay_ms, 1):.1f} fps"
    _put(panel, f"Frame {t + 1}/{n_total}    {status}", (_PAD, 36), color=_GREEN, scale=0.50)

    # ── Task prompt ───────────────────────────────────────────────────────
    short = prompt[:90] + "..." if len(prompt) > 90 else prompt
    _put(panel, f"Task: {short}", (_PAD, 58), color=_YELLOW)

    # ── Observation ───────────────────────────────────────────────────────
    j_obs = "  ".join(f"J{i + 1}:{qpos[i]:+7.2f}" for i in range(6))
    _put(panel, f"Obs  joints [deg]: {j_obs}", (_PAD, 82), color=_WHITE)
    _put(panel, f"Obs  gripper     : {qpos[6]:.3f}  (0=closed  1=open)", (_PAD, 102), color=_WHITE)

    # ── Action ────────────────────────────────────────────────────────────
    j_act = "  ".join(f"J{i + 1}:{action[i]:+7.2f}" for i in range(6))
    _put(panel, f"Cmd  joints [deg]: {j_act}", (_PAD, 126), color=_WHITE)
    _put(panel, f"Cmd  gripper     : {action[6]:.3f}", (_PAD, 146), color=_WHITE)

    # ── Key hints ─────────────────────────────────────────────────────────
    _put(
        panel,
        "[Space] pause/play   [<> / ,.]  step   [+-] speed   [R] restart   [Q/ESC] quit",
        (_PAD, 164),
        scale=0.37,
        color=_GRAY,
    )

    return panel


def build_frame(
    ep: dict,
    t: int,
    n_total: int,
    w_disp: int,
    h_disp: int,
    win_w: int,
    *,
    paused: bool,
    delay_ms: int,
) -> np.ndarray:
    # Camera panels
    ext_bgr = cv2.cvtColor(ep["ext"][t], cv2.COLOR_RGB2BGR)
    wrist_bgr = cv2.cvtColor(ep["wrist"][t], cv2.COLOR_RGB2BGR)
    ext_big = cv2.resize(ext_bgr, (w_disp, h_disp), interpolation=cv2.INTER_NEAREST)
    wrist_big = cv2.resize(wrist_bgr, (w_disp, h_disp), interpolation=cv2.INTER_NEAREST)

    _put(ext_big, "Exterior", (8, 26), scale=0.7, color=_YELLOW, thickness=2)
    _put(wrist_big, "Wrist", (8, 26), scale=0.7, color=_YELLOW, thickness=2)

    pad_v = np.zeros((h_disp, _PAD, 3), dtype=np.uint8)
    cameras = np.hstack([pad_v, ext_big, pad_v, wrist_big, pad_v])

    info = make_info_panel(
        win_w,
        t,
        n_total,
        ep["qpos"][t],
        ep["action"][t],
        ep["prompt"],
        paused=paused,
        delay_ms=delay_ms,
    )
    pad_h = np.zeros((_PAD, win_w, 3), dtype=np.uint8)

    return np.vstack([cameras, pad_h, info])


# ── Playback loop ─────────────────────────────────────────────────────────────


def replay(hdf5_path: Path) -> None:
    print(f"Loading {hdf5_path} ...")
    ep = load_episode(hdf5_path)

    n_total = ep["ext"].shape[0]
    if n_total == 0:
        sys.exit("Episode is empty - nothing to display.")

    print(f"  Steps  : {n_total}")
    print(f"  Hz     : {ep['hz']:.1f}")
    print(f"  Prompt : {ep['prompt'] or '(none)'}")

    h_orig, w_orig = ep["ext"].shape[1:3]  # 224, 224
    h_disp = h_orig * _SCALE  # 448
    w_disp = w_orig * _SCALE  # 448
    win_w = w_disp * 2 + _PAD * 3
    win_h = h_disp + _INFO_H + _PAD

    delay_ms = max(1, int(1000 / ep["hz"]))
    paused = False
    t = 0

    cv2.namedWindow("Replay  [exterior | wrist]", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Replay  [exterior | wrist]", win_w, win_h)

    while True:
        frame = build_frame(ep, t, n_total, w_disp, h_disp, win_w, paused=paused, delay_ms=delay_ms)
        cv2.imshow("Replay  [exterior | wrist]", frame)

        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF

        if key in (27, ord("q"), ord("Q")):  # ESC / Q  -> quit
            break
        if key == ord(" "):  # Space    -> toggle pause
            paused = not paused
        elif key in (83, ord(".")):  # ->  / .   -> step forward
            t = min(t + 1, n_total - 1)
            paused = True
        elif key in (81, ord(",")):  # <-  / ,   -> step backward
            t = max(t - 1, 0)
            paused = True
        elif key in (ord("+"), ord("=")):  # +        -> speed up
            delay_ms = max(1, delay_ms // 2)
        elif key in (ord("-"), ord("_")):  # -        -> slow down
            delay_ms = min(5000, delay_ms * 2)
        elif key in (ord("r"), ord("R")):  # R        -> restart
            t = 0
            paused = False

        # Auto-advance when playing (waitKey timed out -> key == 255)
        if not paused:
            t += 1
            if t >= n_total:
                t = n_total - 1
                paused = True  # auto-pause at end

    cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replay a recorded UR5 HDF5 episode (offline, no ROS required)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "path",
        type=Path,
        help="Path to an episode_N.hdf5 file  OR  a dataset directory",
    )
    p.add_argument(
        "-e",
        "--episode",
        type=int,
        default=0,
        help="Episode index when PATH is a directory",
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    hdf5_path = resolve_path(args.path, args.episode)
    replay(hdf5_path)
