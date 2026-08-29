#!/usr/bin/env bash
# Run NDT demo with simulation, rosbag playback, and recording
# Usage: run_demo.sh [--cuda] <map_dir> <rosbag> <output_dir>

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
AUTOWARE_ACTIVATE="$SCRIPT_DIR/activate_autoware.sh"
COMPARISON_SETUP="$PROJECT_DIR/tests/comparison/install/setup.bash"

# Source NDT topics
source "$SCRIPT_DIR/ndt_topics.sh"

# Parse flags
USE_CUDA=""
INIT_MODE=""
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --cuda)
            USE_CUDA="--cuda"
            shift
            ;;
        --init-mode)
            INIT_MODE="--init-mode"
            shift
            ;;
        *)
            echo "Unknown flag: $1" >&2
            exit 1
            ;;
    esac
done

# Parse positional arguments
MAP_DIR="$1"
ROSBAG="$2"
OUTPUT_DIR="$3"

# Create output directory and generate bag name
mkdir -p "$OUTPUT_DIR"
if [[ -n "$USE_CUDA" ]]; then
    BAG_NAME="$OUTPUT_DIR/cuda_$(date +%Y%m%d_%H%M%S)"
else
    BAG_NAME="$OUTPUT_DIR/builtin_$(date +%Y%m%d_%H%M%S)"
fi

echo "Starting NDT demo..."
echo "  Mode: ${USE_CUDA:-builtin}"
echo "  Init mode: ${INIT_MODE:-disabled}"
echo "  Map: $MAP_DIR"
echo "  Rosbag: $ROSBAG"
echo "  Recording to: $BAG_NAME"

# Export NDT debug environment variables
if [[ -n "$USE_CUDA" ]]; then
    export NDT_DEBUG=1
    export NDT_DEBUG_FILE="${NDT_DEBUG_FILE:-/tmp/ndt_cuda_debug.jsonl}"
    export NDT_DEBUG_VPP=1  # Enable voxel-per-point distribution logging
    echo "CUDA NDT debug enabled: $NDT_DEBUG_FILE (VPP debug on)"
else
    # For builtin mode, use patched Autoware if available
    if [[ -f "$COMPARISON_SETUP" ]]; then
        echo "Using patched Autoware from: tests/comparison/install/"
        export NDT_DEBUG=1
        export NDT_DEBUG_FILE="${NDT_DEBUG_FILE:-/tmp/ndt_autoware_debug.jsonl}"
        echo "Autoware NDT debug enabled: $NDT_DEBUG_FILE"
    fi
fi

# Run simulation, bag play, and recording in parallel
# --halt now,done=1: When any job finishes, kill all others immediately
# This ensures cleanup when ros2 bag play completes (since -l flag removed)
#
# Both the player and the recorder wait on the scan matcher's own readiness
# rather than on a clock. A fixed sleep here was tuned on an x86 desktop and is
# 10x too short on an AGX Orin, where play_launch needs ~55 s before any node
# exists; the bag then played into nothing and --halt tore the run down, which
# presented as the matcher failing to converge rather than as a harness fault.
# See scripts/wait_for_ndt.sh.
#
# The recorder is started BEFORE the player and given a moment to finish
# subscribing. The previous ordering started it 5 s after playback began, which
# silently dropped the opening of every run.
SETTLE="${NDT_STARTUP_DELAY:-3}"

# Clear any readiness flag left by a previous run, or both waiters return at once
# against a stack that is not up yet.
export NDT_READY_FILE="${NDT_READY_FILE:-/tmp/ndt_sim_ready}"
rm -f "$NDT_READY_FILE"

parallel --halt now,done=1 --line-buffer ::: \
    "$SCRIPT_DIR/run_ndt_simulation.sh $USE_CUDA $INIT_MODE '$MAP_DIR'" \
    "$SCRIPT_DIR/wait_for_ndt.sh && sleep $((SETTLE + 2)) && source '$AUTOWARE_ACTIVATE' && ros2 bag play ${BAG_PLAY_ARGS:-} '$ROSBAG'" \
    "$SCRIPT_DIR/wait_for_ndt.sh && sleep $SETTLE && source '$AUTOWARE_ACTIVATE' && ros2 bag record -o '$BAG_NAME' ${NDT_TOPICS[*]}"

echo "Demo finished. Recording saved to: $BAG_NAME"
