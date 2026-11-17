#!/usr/bin/env bash
#
# Activate the Autoware underlay. Sourced by the justfile and the scripts.
#
# The version comes from scripts/autoware_version.sh: AUTOWARE_VERSION, then the
# golf-cart superproject's GOLFCART_AUTOWARE_VERSION or its own selection, then
# 1.5.0. The comparison fork in tests/comparison/autoware_core is
# autoware_core 1.10.0, the core Autoware 2.1.0 ships, so its justfile sets
# AUTOWARE_VERSION=2.1.0.
#

# shellcheck source=autoware_version.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/autoware_version.sh"
if [[ ! -f "$AUTOWARE_PREFIX/setup.bash" ]]; then
    echo "ERROR: Autoware $AUTOWARE_VERSION is not installed ($AUTOWARE_PREFIX/setup.bash not found)" >&2
    return 1 2>/dev/null || exit 1
fi
source "$AUTOWARE_PREFIX/setup.bash"

# Callers outside this repository (the golf-cart superproject's restricted-FOV
# campaign) source this and then <repo>/install/setup.bash. That symlink names
# the version last built, which need not be the one just activated.
_cuda_ndt_link="$CUDA_NDT_ROOT/install"
if [[ -z "${CUDA_NDT_SKIP_LINK_CHECK:-}" && -L "$_cuda_ndt_link" ]] \
   && [[ "$(readlink "$_cuda_ndt_link")" != "$(basename "$CUDA_NDT_INSTALL_DIR")" ]]; then
    echo "WARNING: $_cuda_ndt_link points at $(readlink "$_cuda_ndt_link"), not $(basename "$CUDA_NDT_INSTALL_DIR") (Autoware $AUTOWARE_VERSION); run 'just build'" >&2
fi
unset _cuda_ndt_link
