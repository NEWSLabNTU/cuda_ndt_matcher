#!/bin/bash
#
# Run a command with tegrastats logging
#
# Usage: ./run_with_tegrastats.sh <output_file> <command...>
#
# Starts tegrastats in background, runs the command, then stops tegrastats.
# The output file will contain tegrastats output during the run.

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <output_file> <command...>"
    exit 1
fi

OUTPUT_FILE="$1"
shift

# Kill any existing tegrastats
pkill -9 tegrastats 2>/dev/null || true
sleep 0.5

# Start tegrastats in background
echo "Starting tegrastats logging to: $OUTPUT_FILE"
tegrastats --interval 500 > "$OUTPUT_FILE" 2>&1 &
TEGRA_PID=$!

# Ensure we clean up tegrastats on exit
cleanup() {
    echo "Stopping tegrastats (PID: $TEGRA_PID)"
    kill $TEGRA_PID 2>/dev/null || true
    pkill -9 tegrastats 2>/dev/null || true
}
trap cleanup EXIT

# Wait for tegrastats to start
sleep 1

# Run the actual command
echo "Running: $@"
"$@"

# Give tegrastats a moment to capture final stats
sleep 1

echo "tegrastats log saved to: $OUTPUT_FILE"
