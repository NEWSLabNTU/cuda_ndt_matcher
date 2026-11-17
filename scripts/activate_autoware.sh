#!/usr/bin/env bash
#
# Activate Autoware environment
#
# This script is sourced by other scripts to set up the Autoware environment.
# AUTOWARE_VERSION picks the installation under /opt/autoware (default 1.5.0).
# The comparison fork in tests/comparison/autoware_core is autoware_core 1.10.0,
# the core Autoware 2.1.0 ships, so build it with AUTOWARE_VERSION=2.1.0.
#

source "/opt/autoware/${AUTOWARE_VERSION:-1.5.0}/setup.bash"
