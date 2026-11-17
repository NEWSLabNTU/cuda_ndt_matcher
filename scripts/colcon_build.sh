#!/usr/bin/env bash
#
# colcon build into the selected Autoware version's workspace:
#
#   scripts/colcon_build.sh [colcon build arguments...]
#
# Activates ROS and the Autoware underlay (scripts/activate_autoware.sh), adds
# --build-base build-<ver> and --install-base install-<ver>, and afterwards
# points <repo>/install at install-<ver>, so that callers which source
# <repo>/install/setup.bash get the version just built.
#
set -eo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
# The install link is about to be repointed; no need to warn about it.
export CUDA_NDT_SKIP_LINK_CHECK=1
# shellcheck source=activate_autoware.sh
source "$ROOT/scripts/activate_autoware.sh"

cd "$ROOT"
echo "Building against Autoware $AUTOWARE_VERSION into $(basename "$CUDA_NDT_INSTALL_DIR")/"
colcon build --build-base "$CUDA_NDT_BUILD_DIR" --install-base "$CUDA_NDT_INSTALL_DIR" "$@"

# A real install/ directory is a workspace from before per-version
# directories, built against whichever Autoware was sourced then. Leave it,
# but say so: nothing here uses it any more, and `just clean` removes it.
if [[ -e install && ! -L install ]]; then
    echo "WARNING: install/ is a workspace from before per-version directories; 'just clean' removes it." >&2
    echo "         Until then, callers that source install/setup.bash do not get $(basename "$CUDA_NDT_INSTALL_DIR")/." >&2
else
    ln -sfn "$(basename "$CUDA_NDT_INSTALL_DIR")" install
fi
