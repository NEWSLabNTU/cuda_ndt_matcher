#!/usr/bin/env bash
#
# Activate this repository's colcon workspace for the selected Autoware
# version (install-<ver>/setup.bash, which chains to the Autoware underlay it
# was built against). Sourced by the justfile and the scripts; run
# `just build` first.
#

# shellcheck source=autoware_version.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/autoware_version.sh"
if [[ ! -f "$CUDA_NDT_INSTALL_DIR/setup.bash" ]]; then
    echo "ERROR: no workspace for Autoware $AUTOWARE_VERSION ($CUDA_NDT_INSTALL_DIR); run 'just build'" >&2
    return 1 2>/dev/null || exit 1
fi
source "$CUDA_NDT_INSTALL_DIR/setup.bash"
