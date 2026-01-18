#!/usr/bin/env bash
# Profile NDT implementations and generate comparison report
#
# Usage:
#   ./scripts/profile_ndt.sh [OPTIONS]
#
# Options:
#   --modes <modes>     Comma-separated modes to profile: builtin,cuda-cpu,cuda-gpu
#                       Default: builtin,cuda-gpu
#   --duration <sec>    Profile duration in seconds (default: 60)
#   --perf              Enable perf profiling (requires sudo or CAP_PERFMON)
#   --nsys              Enable nsys GPU profiling (cuda modes only)
#   --output <dir>      Output directory (default: profile_results/<timestamp>)
#   --skip-run          Skip running demos, only analyze existing data
#   --help              Show this help

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
AUTOWARE_SETUP="$PROJECT_DIR/external/autoware_repo/install/setup.bash"
LOCAL_SETUP="$PROJECT_DIR/install/setup.bash"

# Default options
MODES="builtin,cuda-gpu"
DURATION=60
USE_PERF=false
USE_NSYS=false
OUTPUT_DIR=""
SKIP_RUN=false
SAMPLE_MAP_DIR="$PROJECT_DIR/data/sample-map"
SAMPLE_ROSBAG="$PROJECT_DIR/data/sample-rosbag-fixed"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --modes)
            MODES="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --perf)
            USE_PERF=true
            shift
            ;;
        --nsys)
            USE_NSYS=true
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-run)
            SKIP_RUN=true
            shift
            ;;
        --help)
            head -20 "$0" | tail -18
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Setup output directory
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$PROJECT_DIR/profile_results/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR"

echo "=== NDT Profiling ==="
echo "Modes: $MODES"
echo "Duration: ${DURATION}s"
echo "Output: $OUTPUT_DIR"
echo "Perf: $USE_PERF"
echo "Nsys: $USE_NSYS"
echo ""

# Source ROS environment
source "$AUTOWARE_SETUP"
if [[ -f "$LOCAL_SETUP" ]]; then
    source "$LOCAL_SETUP"
fi

# Function to run a single mode
run_mode() {
    local mode="$1"
    local mode_dir="$OUTPUT_DIR/$mode"
    mkdir -p "$mode_dir"

    echo "=== Profiling mode: $mode ==="

    # Set environment variables based on mode
    case "$mode" in
        builtin)
            export NDT_AUTOWARE_DEBUG=1
            export NDT_AUTOWARE_DEBUG_FILE="$mode_dir/debug.jsonl"
            unset NDT_DEBUG NDT_DEBUG_FILE NDT_USE_GPU
            local USE_CUDA=""
            ;;
        cuda-cpu)
            export NDT_DEBUG=1
            export NDT_DEBUG_FILE="$mode_dir/debug.jsonl"
            export NDT_USE_GPU=0
            unset NDT_AUTOWARE_DEBUG NDT_AUTOWARE_DEBUG_FILE
            local USE_CUDA="--cuda"
            ;;
        cuda-gpu)
            export NDT_DEBUG=1
            export NDT_DEBUG_FILE="$mode_dir/debug.jsonl"
            export NDT_USE_GPU=1
            unset NDT_AUTOWARE_DEBUG NDT_AUTOWARE_DEBUG_FILE
            local USE_CUDA="--cuda"
            ;;
        *)
            echo "Unknown mode: $mode"
            return 1
            ;;
    esac

    # Clear previous debug output
    rm -f "$mode_dir/debug.jsonl"

    # Start the simulation in background
    echo "Starting simulation..."
    "$SCRIPT_DIR/run_ndt_simulation.sh" $USE_CUDA "$SAMPLE_MAP_DIR" &
    local SIM_PID=$!

    # Wait for initialization
    echo "Waiting for initialization (30s)..."
    sleep 30

    # Find the NDT process
    local NDT_PROC=""
    if [[ "$mode" == "builtin" ]]; then
        NDT_PROC=$(pgrep -f "ndt_scan_matcher" | head -1 || true)
    else
        NDT_PROC=$(pgrep -f "cuda_ndt_matcher" | head -1 || true)
    fi

    if [[ -z "$NDT_PROC" ]]; then
        echo "Warning: Could not find NDT process"
    else
        echo "Found NDT process: $NDT_PROC"
    fi

    # Start perf profiling if requested
    local PERF_PID=""
    if [[ "$USE_PERF" == "true" && -n "$NDT_PROC" ]]; then
        echo "Starting perf record..."
        sudo perf record -g -F 999 --call-graph dwarf -p "$NDT_PROC" -o "$mode_dir/perf.data" &
        PERF_PID=$!
    fi

    # Start nsys profiling if requested (GPU modes only)
    local NSYS_PID=""
    if [[ "$USE_NSYS" == "true" && "$mode" == "cuda-gpu" && -n "$NDT_PROC" ]]; then
        echo "Starting nsys profile..."
        nsys profile -o "$mode_dir/nsys_profile" --attach "$NDT_PROC" --duration "$DURATION" &
        NSYS_PID=$!
    fi

    # Start rosbag playback
    echo "Starting rosbag playback..."
    ros2 bag play "$SAMPLE_ROSBAG" &
    local BAG_PID=$!

    # Wait for rosbag to finish or timeout
    echo "Profiling for ${DURATION}s..."
    local start_time=$(date +%s)
    while kill -0 "$BAG_PID" 2>/dev/null; do
        local elapsed=$(($(date +%s) - start_time))
        if [[ $elapsed -ge $DURATION ]]; then
            echo "Duration reached, stopping..."
            break
        fi
        sleep 1
    done

    # Stop perf
    if [[ -n "$PERF_PID" ]]; then
        echo "Stopping perf..."
        sudo kill -INT "$PERF_PID" 2>/dev/null || true
        wait "$PERF_PID" 2>/dev/null || true
    fi

    # Wait for nsys to finish
    if [[ -n "$NSYS_PID" ]]; then
        echo "Waiting for nsys..."
        wait "$NSYS_PID" 2>/dev/null || true
    fi

    # Cleanup
    echo "Cleaning up..."
    kill "$BAG_PID" 2>/dev/null || true
    kill "$SIM_PID" 2>/dev/null || true

    # Kill any remaining ROS processes from this mode
    pkill -f "ros2 bag play" 2>/dev/null || true
    if [[ "$mode" == "builtin" ]]; then
        pkill -f "ndt_scan_matcher" 2>/dev/null || true
    else
        pkill -f "cuda_ndt_matcher" 2>/dev/null || true
    fi

    # Wait for cleanup
    sleep 5

    # Generate perf report if we have data
    if [[ -f "$mode_dir/perf.data" ]]; then
        echo "Generating perf report..."
        sudo perf report -i "$mode_dir/perf.data" --stdio > "$mode_dir/perf_report.txt" 2>/dev/null || true

        # Generate flamegraph if tools available
        if command -v stackcollapse-perf.pl &>/dev/null && command -v flamegraph.pl &>/dev/null; then
            echo "Generating flamegraph..."
            sudo perf script -i "$mode_dir/perf.data" 2>/dev/null | \
                stackcollapse-perf.pl | \
                flamegraph.pl > "$mode_dir/flamegraph.svg" 2>/dev/null || true
        fi
    fi

    echo "Mode $mode complete. Results in $mode_dir"
    echo ""
}

# Run each mode unless --skip-run
if [[ "$SKIP_RUN" == "false" ]]; then
    IFS=',' read -ra MODE_ARRAY <<< "$MODES"
    for mode in "${MODE_ARRAY[@]}"; do
        run_mode "$mode"
    done
fi

# Run analysis
echo "=== Running Analysis ==="
python3 "$SCRIPT_DIR/analyze_profile.py" "$OUTPUT_DIR"

echo ""
echo "=== Profiling Complete ==="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files generated:"
ls -la "$OUTPUT_DIR"/*/ 2>/dev/null || true
echo ""
echo "Summary report: $OUTPUT_DIR/summary.txt"
