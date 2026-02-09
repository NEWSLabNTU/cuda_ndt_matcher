#!/usr/bin/env bash
# Run NDT replay simulation
# Usage: run_ndt_simulation.sh [--cuda] [--no-rviz] <map_path>

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
AUTOWARE_ACTIVATE="$SCRIPT_DIR/activate_autoware.sh"
COMPARISON_SETUP="$PROJECT_DIR/tests/comparison/install/setup.bash"

USE_CUDA="false"
RVIZ="true"
INIT_MODE="false"
MAP_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            USE_CUDA="true"
            shift
            ;;
        --no-rviz)
            RVIZ="false"
            shift
            ;;
        --init-mode)
            INIT_MODE="true"
            shift
            ;;
        *)
            MAP_PATH="$1"
            shift
            ;;
    esac
done

if [[ -z "$MAP_PATH" ]]; then
    echo "Usage: $0 [--cuda] [--no-rviz] <map_path>" >&2
    exit 1
fi

# Auto-detect rviz availability if not explicitly disabled
if [[ "$RVIZ" == "true" ]]; then
    if [[ -z "$DISPLAY" ]] || ! xdpyinfo &>/dev/null; then
        RVIZ="false"
    fi
fi

# Enable user-defined initial pose for demo runs by default
# This provides a consistent starting pose for reproducible testing
# Without this, the EKF initializes to an unknown state
# Use --init-mode to disable this and test Monte Carlo pose init
if [[ "$INIT_MODE" == "true" ]]; then
    USE_INITIAL_POSE="false"
    echo "Init mode: Monte Carlo pose initialization enabled"
else
    USE_INITIAL_POSE="true"
fi

# Ensure rosbag replay sensor kit packages are installed
if [[ ! -d "$PROJECT_DIR/install/rosbag_sensor_kit_launch" ]]; then
    echo "Building rosbag replay sensor kit packages..."
    (cd "$PROJECT_DIR" && colcon build \
        --base-paths tests/rosbag_replay/individual_params \
                     tests/rosbag_replay/rosbag_sensor_kit_description \
                     tests/rosbag_replay/rosbag_sensor_kit_launch \
        --symlink-install \
        --cmake-args -DCMAKE_BUILD_TYPE=Release)
fi

# Source Autoware environment first, then local workspace as overlay
source "$AUTOWARE_ACTIVATE"
source "$PROJECT_DIR/install/setup.bash"

# For builtin mode, overlay patched Autoware if available
if [[ "$USE_CUDA" == "false" && -f "$COMPARISON_SETUP" ]]; then
    echo "Overlaying patched Autoware from: tests/comparison/install/"
    source "$COMPARISON_SETUP"
fi

# Export NDT_DEBUG if CUDA mode and not already set
if [[ "$USE_CUDA" == "true" && -z "${NDT_DEBUG:-}" ]]; then
    export NDT_DEBUG=1
    export NDT_DEBUG_FILE="${NDT_DEBUG_FILE:-/tmp/ndt_cuda_debug.jsonl}"
    export NDT_DEBUG_VPP=1  # Enable voxel-per-point distribution logging
    echo "CUDA NDT debug enabled: $NDT_DEBUG_FILE (VPP debug on)"
fi

# Export NDT_AUTOWARE_DEBUG if builtin mode and not already set
if [[ "$USE_CUDA" == "false" && -z "${NDT_AUTOWARE_DEBUG:-}" ]]; then
    export NDT_AUTOWARE_DEBUG=1
    export NDT_AUTOWARE_DEBUG_FILE="${NDT_AUTOWARE_DEBUG_FILE:-/tmp/ndt_autoware_debug.jsonl}"
    echo "Autoware NDT debug enabled: $NDT_AUTOWARE_DEBUG_FILE"
fi

exec \
    play_launch launch \
    --web-addr 0.0.0.0:8888 \
    cuda_ndt_matcher_launch ndt_replay_simulation.launch.xml \
    use_cuda:="$USE_CUDA" \
    map_path:="$MAP_PATH" \
    rviz:="$RVIZ" \
    user_defined_initial_pose_enable:="$USE_INITIAL_POSE"
