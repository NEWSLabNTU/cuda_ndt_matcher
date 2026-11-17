# shellcheck shell=bash
# autoware_version.sh - which Autoware release this checkout builds and runs
# against, and where its colcon workspace for that release lives.
#
# Sourced, never run: by activate_autoware.sh, activate_workspace.sh,
# colcon_build.sh and the justfile. Resolution, first hit wins:
#
#   1. AUTOWARE_VERSION, if set: a one-off override (tests/comparison sets
#      it), and also what Autoware's own setup.bash exports, so a shell that
#      already sourced an Autoware keeps that one (two underlays cannot mix)
#   2. GOLFCART_AUTOWARE_VERSION, if set (exported by the golf-cart
#      superproject's scripts/env.sh)
#   3. the superproject's own selection, when this checkout is its submodule
#      and it has scripts/autoware_version.sh (its config/autoware-version,
#      then its default), so a shell that never sourced its env.sh agrees
#      with one that did
#   4. 1.5.0
#
# Exports:
#   AUTOWARE_VERSION       e.g. 2.1.0
#   AUTOWARE_PREFIX        /opt/autoware/<ver>
#   CUDA_NDT_BUILD_DIR     <repo>/build-<ver>
#   CUDA_NDT_INSTALL_DIR   <repo>/install-<ver>
#
# Colcon workspaces cannot be shared between underlays, so every version gets
# its own build and install directory. <repo>/install is a symlink to the
# install directory of the version last built; activate_workspace.sh refuses
# it when it points at another version than the one selected.

: "${CUDA_NDT_ROOT:=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"

cuda_ndt_resolve_autoware_version() {
    local root="${CUDA_NDT_ROOT}" v="${AUTOWARE_VERSION:-${GOLFCART_AUTOWARE_VERSION:-}}" sp
    if [ -z "$v" ]; then
        sp="$(git -C "$root" rev-parse --show-superproject-working-tree 2>/dev/null || true)"
        if [ -n "$sp" ] && [ -f "$sp/scripts/autoware_version.sh" ]; then
            # In a child shell: the superproject's script exports a set of
            # GOLFCART_* variables this repo has no business setting.
            v="$(GOLFCART_REPO_ROOT="$sp" bash -c \
                '. "$1/scripts/autoware_version.sh" >/dev/null 2>&1 && printf %s "$GOLFCART_AUTOWARE_VERSION"' \
                _ "$sp" 2>/dev/null || true)"
        fi
    fi
    if [ -n "${AUTOWARE_VERSION:-}" ] && [ -n "${GOLFCART_AUTOWARE_VERSION:-}" ] \
       && [ "${AUTOWARE_VERSION}" != "${GOLFCART_AUTOWARE_VERSION}" ]; then
        echo "WARNING: AUTOWARE_VERSION=${AUTOWARE_VERSION} (an Autoware already sourced, or an override) wins over GOLFCART_AUTOWARE_VERSION=${GOLFCART_AUTOWARE_VERSION}; open a new shell to switch." >&2
    fi
    v="${v:-1.5.0}"
    if ! [[ "$v" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "WARNING: Autoware version '$v' is not X.Y.Z; using 1.5.0." >&2
        v=1.5.0
    fi
    AUTOWARE_VERSION="$v"
    AUTOWARE_PREFIX="/opt/autoware/$v"
    CUDA_NDT_BUILD_DIR="$root/build-$v"
    CUDA_NDT_INSTALL_DIR="$root/install-$v"
    export AUTOWARE_VERSION AUTOWARE_PREFIX CUDA_NDT_BUILD_DIR CUDA_NDT_INSTALL_DIR
}

cuda_ndt_resolve_autoware_version
