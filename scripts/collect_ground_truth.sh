#!/usr/bin/env bash
# Collect Autoware builtin NDT output as ground truth for CUDA NDT validation.
#
# This script runs the builtin NDT demo and saves its output to tests/fixtures/ground_truth/
# Run this once to establish baseline, or whenever the reference implementation changes.
#
# Usage: ./scripts/collect_ground_truth.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GROUND_TRUTH_DIR="$PROJECT_DIR/tests/fixtures/ground_truth"

echo "=== Collecting Ground Truth Data ==="
echo "Output directory: $GROUND_TRUTH_DIR"

# Clean previous ground truth
rm -rf "$GROUND_TRUTH_DIR"
mkdir -p "$GROUND_TRUTH_DIR"

# Run builtin NDT demo
echo ""
echo "Running Autoware builtin NDT demo..."
cd "$PROJECT_DIR"
just run-builtin

# Find the latest builtin rosbag
LATEST_BAG=$(ls -td rosbag/builtin_* 2>/dev/null | head -1)
if [[ -z "$LATEST_BAG" ]]; then
    echo "ERROR: No builtin rosbag found in rosbag/"
    exit 1
fi

# Copy results to fixtures
echo ""
echo "Copying results to ground truth directory..."
cp -r "$LATEST_BAG" "$GROUND_TRUTH_DIR/rosbag"

# Copy debug log if exists
DEBUG_LOG="/tmp/ndt_autoware_debug.jsonl"
if [[ -f "$DEBUG_LOG" ]]; then
    cp "$DEBUG_LOG" "$GROUND_TRUTH_DIR/debug.jsonl"
    echo "  - Copied debug log"
else
    echo "  - No debug log found (this is OK for builtin)"
fi

# Save metadata
cat > "$GROUND_TRUTH_DIR/metadata.json" << EOF
{
    "collected_at": "$(date -Iseconds)",
    "source_rosbag": "$LATEST_BAG",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
}
EOF

echo ""
echo "=== Ground Truth Collection Complete ==="
echo "Saved to: $GROUND_TRUTH_DIR"
echo ""
echo "Contents:"
ls -la "$GROUND_TRUTH_DIR"
echo ""
echo "Rosbag info:"
source "$PROJECT_DIR/external/autoware_repo/install/setup.bash"
ros2 bag info "$GROUND_TRUTH_DIR/rosbag"
