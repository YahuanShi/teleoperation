#!/bin/bash
# One-shot setup for a new PC.
# Usage: bash install.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. ROS 2 Humble (apt) ────────────────────────────────────────────────────
echo "[1/4] Installing ROS 2 Humble apt packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    ros-humble-desktop \
    python3-colcon-common-extensions \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs

# ── 2. Python venv ────────────────────────────────────────────────────────────
echo "[2/4] Creating Python venv at .venv ..."
python3 -m venv "$SCRIPT_DIR/.venv"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.venv/bin/activate"

pip install --upgrade pip -q
pip install -r "$SCRIPT_DIR/uarm/requirements.txt"

# ── 3. Build ROS 2 package ────────────────────────────────────────────────────
echo "[3/4] Building uarm ROS 2 package..."
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
cd "$SCRIPT_DIR"
colcon build --packages-select uarm

# ── 4. Serial port permissions ────────────────────────────────────────────────
echo "[4/4] Adding $USER to dialout group (re-login required)..."
sudo usermod -aG dialout "$USER"

echo ""
echo "Setup complete. Next steps:"
echo "  1. Re-login (or run: newgrp dialout) for serial port access."
echo "  2. Source the workspace: source $SCRIPT_DIR/install/setup.bash"
echo "  3. Auto-activate (direnv): cd $SCRIPT_DIR && direnv allow"
