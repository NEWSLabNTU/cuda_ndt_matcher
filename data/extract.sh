#!/usr/bin/env bash
# Extract split zstd tarballs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

extract_split_tarball() {
    local name="$1"
    if [[ -d "$name" ]]; then
        echo "Directory $name already exists, skipping"
        return
    fi
    if [[ ! -f "$name.tar.zst.aa" ]]; then
        echo "Split files for $name not found, skipping"
        return
    fi
    echo "Extracting $name..."
    cat "$name.tar.zst."* | zstd -d | tar xf -
    echo "Done: $name"
}

extract_split_tarball "sample-rosbag-fixed"
extract_split_tarball "sample-rosbag-original"
