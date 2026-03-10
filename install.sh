#!/bin/bash
# One-shot setup for a new PC.
# Usage: bash install.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. ROS 2 Humble (apt) ────────────────────────────────────────────────────
echo "[1/5] Installing ROS 2 Humble apt packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    ros-humble-desktop \
    python3-colcon-common-extensions \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs

# ── 2. Python venv ────────────────────────────────────────────────────────────
echo "[2/5] Creating Python venv at teleop/ ..."
python3 -m venv "$SCRIPT_DIR/teleop"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/teleop/bin/activate"

pip install --upgrade pip -q
pip install -r "$SCRIPT_DIR/requirements.txt"

# ── 3. Build ROS 2 package ────────────────────────────────────────────────────
echo "[3/5] Building uarm ROS 2 package..."
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
cd "$SCRIPT_DIR"
colcon build --packages-select uarm

# ── 4. Serial port permissions ────────────────────────────────────────────────
echo "[4/5] Adding $USER to dialout group (re-login required)..."
sudo usermod -aG dialout "$USER"

# ── 5. Shell prompt hook (direnv cannot export PS1 directly) ─────────────────
echo "[5/5] Adding virtualenv prompt hook to shell rc..."
HOOK='
# Show active direnv virtualenv in prompt (added by teleoperation/install.sh)
_direnv_venv_ps1() { [[ -n "$VIRTUAL_ENV" && -n "$DIRENV_DIR" ]] && echo "($(basename "$VIRTUAL_ENV")) "; }
PS1='"'"'$(_direnv_venv_ps1)'"'"'$PS1'

SHELL_RC=""
if [[ "$SHELL" == */zsh ]]; then
    SHELL_RC="$HOME/.zshrc"
else
    SHELL_RC="$HOME/.bashrc"
fi

if ! grep -q "_direnv_venv_ps1" "$SHELL_RC" 2>/dev/null; then
    echo "$HOOK" >> "$SHELL_RC"
    echo "    Added prompt hook to $SHELL_RC"
else
    echo "    Prompt hook already present in $SHELL_RC"
fi

echo ""
echo "Setup complete. Next steps:"
echo "  1. Re-login (or run: newgrp dialout) for serial port access."
echo "  2. Reload shell:  source $SHELL_RC"
echo "  3. Allow direnv:  cd $SCRIPT_DIR && direnv allow"
echo "  cd into teleoperation/ will now show (teleop) in the prompt."
