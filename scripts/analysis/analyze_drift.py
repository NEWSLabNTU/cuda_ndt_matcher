#!/usr/bin/env python3
"""Analyze drift patterns in CUDA NDT debug output."""

import json
import sys

def main():
    debug_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ndt_cuda_debug.jsonl"

    with open(debug_file) as f:
        alignments = [json.loads(line) for line in f if line.strip()]

    # Analyze direction reversal patterns
    print("=== Direction Reversal Analysis ===")
    for idx in [95, 96, 97, 98, 99, 100, 101, 102]:
        a = alignments[idx]
        reversals = sum(1 for it in a['iterations'] if it['direction_reversed'])
        total = len(a['iterations'])
        status = "MAX" if a['convergence_status'] == 'MaxIterations' else "OK "
        print(f"Frame {idx}: [{status}] reversals={reversals}/{total} ({100*reversals/total:.0f}%)")

    print("\n=== Initial Pose Source Analysis ===")
    print("Comparing: EKF initial guess vs previous NDT result")
    print()
    for idx in range(95, 105):
        if idx == 0:
            continue
        prev_final = alignments[idx-1]['final_pose']
        curr_init = alignments[idx]['initial_pose']

        delta_x = curr_init[0] - prev_final[0]
        delta_y = curr_init[1] - prev_final[1]
        delta_dist = (delta_x**2 + delta_y**2)**0.5

        status = "MAX" if alignments[idx]['convergence_status'] == 'MaxIterations' else "OK "
        print(f"Frame {idx}: [{status}] EKF drift from prev NDT: {delta_dist:.3f}m (dx={delta_x:+.3f}, dy={delta_y:+.3f})")


if __name__ == "__main__":
    main()
