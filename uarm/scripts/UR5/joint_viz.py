#!/usr/bin/env python3
"""
UR5 Joint + Gripper Trajectory Visualizer

Subscribes to:
    /servo_angles   — U-Arm master arm angles (7 floats, deg)
    /robot_state    — UR5 actual state (6 joint deg + gripper 0-1)
    /is_recording   — Bool from episode_recorder

Layout: 4 rows x 2 cols (8 slots, 7 used)
    J0  J1
    J2  J3
    J4  J5
    Gripper  [empty]

Blue = U-Arm master,  Red = UR5 actual
Trajectories recorded only while /is_recording is True.
"""

from collections import deque
import threading
import time

import matplotlib as mpl

mpl.use("TkAgg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray

# ── Config ────────────────────────────────────────────────────────────────────
JOINT_MAP = [0, 1, 2, 3, 5, 4]  # must match servo2ur5.py
# 7 channels: joints 0-5, then gripper
CHANNEL_NAMES = ["J0", "J1", "J2", "J3", "J4", "J5", "Gripper"]
CHANNEL_YLABELS = ["deg"] * 6 + ["%"]

MAX_SAMPLES = 3000
SAMPLE_HZ = 30
UPDATE_HZ = 15
WINDOW_S = 30.0

WIN_X, WIN_Y = 30, 30
FIG_W, FIG_H = 16, 14

# ── ROS 2 data node ───────────────────────────────────────────────────────────


class TrajVizNode(Node):
    def __init__(self):
        super().__init__("joint_traj_viz_node")
        self._lock = threading.Lock()

        self._is_recording = False
        self._t_start = None

        self._t_buf = deque(maxlen=MAX_SAMPLES)
        self._uarm_buf = [deque(maxlen=MAX_SAMPLES) for _ in range(7)]
        self._ur5_buf = [deque(maxlen=MAX_SAMPLES) for _ in range(7)]

        self._servo = [0.0] * 7
        self._state = [0.0] * 7

        self.create_subscription(Float64MultiArray, "/servo_angles", self._cb_servo, 10)
        self.create_subscription(Float64MultiArray, "/robot_state", self._cb_state, 10)
        self.create_subscription(Bool, "/is_recording", self._cb_rec, 10)
        self.create_timer(1.0 / SAMPLE_HZ, self._sample)

        self.get_logger().info("[TrajViz] Ready")

    def _cb_servo(self, msg):
        if len(msg.data) >= 7:
            with self._lock:
                self._servo = list(msg.data[:7])

    def _cb_state(self, msg):
        if len(msg.data) >= 7:
            with self._lock:
                self._state = list(msg.data[:7])

    def _cb_rec(self, msg):
        with self._lock:
            was = self._is_recording
            self._is_recording = msg.data
            if msg.data and not was:
                self._t_buf.clear()
                for b in self._uarm_buf:
                    b.clear()
                for b in self._ur5_buf:
                    b.clear()
                self._t_start = time.time()

    def _sample(self):
        with self._lock:
            if not self._is_recording:
                return
            t = time.time() - (self._t_start or time.time())
            self._t_buf.append(t)
            for i in range(6):
                self._uarm_buf[i].append(self._servo[JOINT_MAP[i]])
                self._ur5_buf[i].append(self._state[i])
            # gripper: servo[6] in deg, state[6] normalised 0-1 → scale to %
            self._uarm_buf[6].append(self._servo[6])
            self._ur5_buf[6].append(self._state[6] * 100.0)

    def get_data(self):
        with self._lock:
            return (
                list(self._t_buf),
                [list(b) for b in self._uarm_buf],
                [list(b) for b in self._ur5_buf],
                self._is_recording,
                list(self._servo),
                list(self._state),
            )


# ── Plot ──────────────────────────────────────────────────────────────────────


def run_plot(node: TrajVizNode):
    plt.rcParams.update(
        {
            "figure.facecolor": "#1c1c1c",
            "axes.facecolor": "#111111",
            "axes.edgecolor": "#3a3a3a",
            "axes.labelcolor": "#888888",
            "xtick.color": "#666666",
            "ytick.color": "#666666",
            "grid.color": "#2a2a2a",
            "grid.linewidth": 0.8,
            "text.color": "#cccccc",
            "legend.facecolor": "#1a1a1a",
            "legend.edgecolor": "#3a3a3a",
        }
    )

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.30, left=0.07, right=0.97, top=0.93, bottom=0.06)

    # 7 subplots: rows 0-2 both columns, row 3 left only
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]
    axes.append(fig.add_subplot(gs[3, 0]))  # gripper

    # Position window
    try:
        mgr = plt.get_current_fig_manager()
        mgr.window.wm_geometry(f"+{WIN_X}+{WIN_Y}")
    except Exception:
        pass

    lines_uarm, lines_ur5, vlines, val_texts = [], [], [], []

    for i, ax in enumerate(axes):
        ylabel = CHANNEL_YLABELS[i]
        ax.set_title(CHANNEL_NAMES[i], fontsize=11, color="#aaaaaa", pad=4, fontweight="bold")
        ax.set_xlabel("t  (s)", fontsize=8, labelpad=2)
        ax.set_ylabel(ylabel, fontsize=8, labelpad=2)
        ax.tick_params(labelsize=8, length=4)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(which="major", alpha=0.6)
        ax.grid(which="minor", alpha=0.2)
        ax.axhline(0, color="#404040", lw=0.8, ls="--")

        label_u = "U-Arm" if i < 6 else "Master (deg)"
        label_r = "UR5" if i < 6 else "UR5 (%)"
        (lu,) = ax.plot([], [], color="#4488ff", lw=1.8, label=label_u, zorder=3)
        (lr,) = ax.plot([], [], color="#ff4444", lw=1.8, label=label_r, zorder=3)
        vl = ax.axvline(x=0, color="#ffcc00", lw=1.0, ls=":", alpha=0, zorder=4)
        lines_uarm.append(lu)
        lines_ur5.append(lr)
        vlines.append(vl)

        ax.legend(fontsize=8, loc="upper left", ncol=2, handlelength=1.4, columnspacing=0.8, borderpad=0.5)

        vt = ax.text(
            0.99,
            0.97,
            "",
            transform=ax.transAxes,
            fontsize=8,
            color="#ffcc44",
            ha="right",
            va="top",
            fontfamily="monospace",
        )
        val_texts.append(vt)

    title_text = fig.text(
        0.5,
        0.965,
        "Joint & Gripper Trajectories   |   Blue: U-Arm    Red: UR5",
        ha="center",
        fontsize=12,
        color="#cccccc",
        fontweight="bold",
    )
    status_text = fig.text(
        0.5,
        0.012,
        "WAITING  —  press B in the recorder window to start recording",
        ha="center",
        fontsize=9,
        color="#555577",
    )

    plt.ion()
    plt.show()

    dt = 1.0 / UPDATE_HZ

    while rclpy.ok() and plt.fignum_exists(fig.number):
        t_arr, uarm_data, ur5_data, rec, cur_servo, cur_state = node.get_data()
        n = len(t_arr)
        pts = 0

        for i in range(7):
            pts = min(n, len(uarm_data[i]), len(ur5_data[i]))
            if pts > 1:
                t_p = t_arr[:pts]
                lines_uarm[i].set_data(t_p, uarm_data[i][:pts])
                lines_ur5[i].set_data(t_p, ur5_data[i][:pts])
            else:
                lines_uarm[i].set_data([], [])
                lines_ur5[i].set_data([], [])

            # Live current-value label
            if i < 6:
                val_texts[i].set_text(f"U:{cur_servo[JOINT_MAP[i]]:+.1f}°  R:{cur_state[i]:+.1f}°")
            else:
                val_texts[i].set_text(f"U:{cur_servo[6]:+.1f}°  R:{cur_state[6] * 100:.0f}%")

            # "Now" cursor
            if rec and t_arr:
                t_now = t_arr[-1]
                vlines[i].set_xdata([t_now, t_now])
                vlines[i].set_alpha(0.7)
            else:
                vlines[i].set_alpha(0)

        # Axis limits — rolling window during recording, full view after
        if pts > 1:
            if rec:
                t_now = t_arr[-1]
                t_lo = max(0.0, t_now - WINDOW_S)
                for ax in axes:
                    ax.set_xlim(t_lo, t_now + 0.5)
                    ax.relim()
                    ax.autoscale_view(scalex=False, scaley=True, tight=True)
            else:
                for ax in axes:
                    ax.relim()
                    ax.autoscale_view(tight=True)

        # Status bar
        if rec:
            dur = t_arr[-1] if t_arr else 0.0
            status_text.set_text(f"RECORDING   {n} samples   {dur:.1f} s")
            status_text.set_color("#ff6666")
            title_text.set_color("#ff9999")
        else:
            if n > 1:
                dur = t_arr[-1]
                status_text.set_text(f"WAITING  —  last episode: {n} samples  {dur:.1f} s")
            else:
                status_text.set_text("WAITING  —  press B in the recorder window to start recording")
            status_text.set_color("#556688")
            title_text.set_color("#cccccc")

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(dt)

    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    rclpy.init()
    node = TrajVizNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        run_plot(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
