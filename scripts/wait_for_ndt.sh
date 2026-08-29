#!/usr/bin/env bash
# Block until the scan matcher is up, or fail loudly.
#
# Replaces a fixed `sleep`, which was calibrated on an x86 desktop and is not
# portable. play_launch parses the launch tree before it runs anything, and on
# an AGX Orin that parse alone takes ~45 s, with nodes ready at ~55 s. A 5 s
# sleep meant the bag played into a stack that did not exist yet, and
# `parallel --halt now,done=1` then tore everything down when playback ended.
# The run still exited 0 and still wrote a rosbag -- an empty one -- so the
# failure looked like the scan matcher never converging.
#
# Readiness comes from play_launch's own "Startup complete" line, which
# run_ndt_simulation.sh turns into a file. Probing over DDS does not work here:
# `ros2 topic list` answers from the ROS 2 daemon, which this harness starts
# under a different environment than the nodes, so it reports nothing however
# long it is polled.
set -eo pipefail

READY_FILE="${NDT_READY_FILE:-/tmp/ndt_sim_ready}"
TIMEOUT="${NDT_STARTUP_TIMEOUT:-300}"

deadline=$((SECONDS + TIMEOUT))
while (( SECONDS < deadline )); do
    if [[ -e "$READY_FILE" ]]; then
        echo "wait_for_ndt: stack ready after ${SECONDS}s"
        exit 0
    fi
    sleep 1
done

echo "wait_for_ndt: stack not ready within ${TIMEOUT}s (no $READY_FILE)" >&2
exit 1
