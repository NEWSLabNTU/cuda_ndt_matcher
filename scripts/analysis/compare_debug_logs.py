#!/usr/bin/env python3
"""Compare CUDA NDT and Autoware NDT debug logs side by side.

This script reads the JSON lines debug files from both implementations
and compares them iteration by iteration to identify divergence points.

Usage:
    python3 scripts/analysis/compare_debug_logs.py [autoware_log] [cuda_log]

Defaults:
    autoware_log: /tmp/ndt_autoware_debug.jsonl
    cuda_log: /tmp/ndt_cuda_debug.jsonl
"""

import json
import sys
from pathlib import Path
import math


def load_debug_log(path):
    """Load debug log file and return list of alignment records."""
    alignments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    alignments.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line in {path}: {e}")
    return alignments


def pose_distance(p1, p2):
    """Euclidean distance between two poses (XY only)."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def pose_angle_diff(p1, p2):
    """Angular difference in yaw (index 5)."""
    diff = abs(p1[5] - p2[5])
    # Normalize to [0, pi]
    while diff > math.pi:
        diff = abs(diff - 2 * math.pi)
    return diff


def compare_alignments(autoware, cuda, idx):
    """Compare a single alignment between Autoware and CUDA."""
    print(f"\n{'='*60}")
    print(f"Frame {idx}")
    print(f"{'='*60}")

    # Initial pose comparison
    aw_init = autoware.get('initial_pose', [0]*6)
    cu_init = cuda.get('initial_pose', [0]*6)
    init_dist = pose_distance(aw_init, cu_init)
    init_angle = pose_angle_diff(aw_init, cu_init)

    print(f"\nInitial Pose:")
    print(f"  Autoware: ({aw_init[0]:.4f}, {aw_init[1]:.4f}, {aw_init[2]:.4f})")
    print(f"      CUDA: ({cu_init[0]:.4f}, {cu_init[1]:.4f}, {cu_init[2]:.4f})")
    print(f"  Distance: {init_dist:.6f}m, Yaw diff: {math.degrees(init_angle):.4f} deg")

    # Final pose comparison
    aw_final = autoware.get('final_pose', [0]*6)
    cu_final = cuda.get('final_pose', [0]*6)
    final_dist = pose_distance(aw_final, cu_final)
    final_angle = pose_angle_diff(aw_final, cu_final)

    print(f"\nFinal Pose:")
    print(f"  Autoware: ({aw_final[0]:.4f}, {aw_final[1]:.4f}, {aw_final[2]:.4f})")
    print(f"      CUDA: ({cu_final[0]:.4f}, {cu_final[1]:.4f}, {cu_final[2]:.4f})")
    print(f"  Distance: {final_dist:.6f}m, Yaw diff: {math.degrees(final_angle):.4f} deg")

    # Convergence comparison
    aw_iters = autoware.get('total_iterations', 0)
    cu_iters = cuda.get('total_iterations', 0)
    aw_status = autoware.get('convergence_status', 'Unknown')
    cu_status = cuda.get('convergence_status', 'Unknown')

    print(f"\nConvergence:")
    print(f"  Autoware: {aw_status} in {aw_iters} iterations")
    print(f"      CUDA: {cu_status} in {cu_iters} iterations")

    # Score comparison
    aw_score = autoware.get('final_score', 0)
    cu_score = cuda.get('final_score', 0)
    aw_nvtl = autoware.get('final_nvtl', 0)
    cu_nvtl = cuda.get('final_nvtl', 0)

    print(f"\nScoring:")
    print(f"  Autoware: score={aw_score:.2f}, NVTL={aw_nvtl:.4f}")
    print(f"      CUDA: score={cu_score:.2f}, NVTL={cu_nvtl:.4f}")

    # Per-iteration comparison if available
    aw_iters_data = autoware.get('iterations', [])
    cu_iters_data = cuda.get('iterations', [])

    if aw_iters_data and cu_iters_data:
        print(f"\nPer-iteration comparison:")
        print(f"  Iter |     Autoware Pose     |        CUDA Pose      | Pose Diff | Score Diff")
        print(f"  {'-'*80}")

        max_iters = max(len(aw_iters_data), len(cu_iters_data))
        for i in range(min(max_iters, 10)):  # Show first 10 iterations
            aw_it = aw_iters_data[i] if i < len(aw_iters_data) else None
            cu_it = cu_iters_data[i] if i < len(cu_iters_data) else None

            if aw_it and cu_it:
                aw_p = aw_it.get('pose', [0]*6)
                cu_p = cu_it.get('pose', [0]*6)
                p_dist = pose_distance(aw_p, cu_p)
                s_diff = cu_it.get('score', 0) - aw_it.get('score', 0)

                aw_rev = " R" if aw_it.get('direction_reversed', False) else "  "
                cu_rev = " R" if cu_it.get('direction_reversed', False) else "  "

                print(f"  {i:4d} | ({aw_p[0]:9.3f},{aw_p[1]:9.3f}){aw_rev} | "
                      f"({cu_p[0]:9.3f},{cu_p[1]:9.3f}){cu_rev} | {p_dist:9.6f} | {s_diff:+10.2f}")
            elif aw_it:
                aw_p = aw_it.get('pose', [0]*6)
                print(f"  {i:4d} | ({aw_p[0]:9.3f},{aw_p[1]:9.3f})   |        (CUDA ended)        | -")
            elif cu_it:
                cu_p = cu_it.get('pose', [0]*6)
                print(f"  {i:4d} |      (Autoware ended)      | ({cu_p[0]:9.3f},{cu_p[1]:9.3f})   | -")

    return {
        'frame': idx,
        'init_dist': init_dist,
        'final_dist': final_dist,
        'aw_iters': aw_iters,
        'cu_iters': cu_iters,
        'aw_status': aw_status,
        'cu_status': cu_status,
    }


def find_divergence_point(autoware_logs, cuda_logs, threshold=0.1):
    """Find the first frame where final poses diverge by more than threshold."""
    min_len = min(len(autoware_logs), len(cuda_logs))

    for i in range(min_len):
        aw_final = autoware_logs[i].get('final_pose', [0]*6)
        cu_final = cuda_logs[i].get('final_pose', [0]*6)
        dist = pose_distance(aw_final, cu_final)

        if dist > threshold:
            return i, dist

    return None, 0


def main():
    autoware_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/ndt_autoware_debug.jsonl'
    cuda_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/ndt_cuda_debug.jsonl'

    print(f"Loading debug logs...")
    print(f"  Autoware: {autoware_path}")
    print(f"      CUDA: {cuda_path}")

    if not Path(autoware_path).exists():
        print(f"Error: Autoware debug log not found: {autoware_path}")
        print("Run 'just run-builtin' first to generate Autoware logs")
        return 1

    if not Path(cuda_path).exists():
        print(f"Error: CUDA debug log not found: {cuda_path}")
        print("Run 'just run-cuda' first to generate CUDA logs")
        return 1

    autoware_logs = load_debug_log(autoware_path)
    cuda_logs = load_debug_log(cuda_path)

    print(f"\nLoaded {len(autoware_logs)} Autoware alignments")
    print(f"Loaded {len(cuda_logs)} CUDA alignments")

    # Find divergence point
    div_frame, div_dist = find_divergence_point(autoware_logs, cuda_logs, threshold=0.1)

    if div_frame is not None:
        print(f"\n*** Divergence detected at frame {div_frame} (distance: {div_dist:.4f}m) ***")

        # Show frames around divergence point
        start_frame = max(0, div_frame - 2)
        end_frame = min(len(autoware_logs), len(cuda_logs), div_frame + 3)

        for i in range(start_frame, end_frame):
            compare_alignments(autoware_logs[i], cuda_logs[i], i)
    else:
        print("\nNo significant divergence detected (threshold: 0.1m)")

        # Show summary statistics
        print("\n=== Summary ===")
        min_len = min(len(autoware_logs), len(cuda_logs))

        total_dist = 0
        aw_total_iters = 0
        cu_total_iters = 0

        for i in range(min_len):
            aw_final = autoware_logs[i].get('final_pose', [0]*6)
            cu_final = cuda_logs[i].get('final_pose', [0]*6)
            total_dist += pose_distance(aw_final, cu_final)
            aw_total_iters += autoware_logs[i].get('total_iterations', 0)
            cu_total_iters += cuda_logs[i].get('total_iterations', 0)

        print(f"Average final pose distance: {total_dist/min_len:.6f}m")
        print(f"Average iterations - Autoware: {aw_total_iters/min_len:.1f}, CUDA: {cu_total_iters/min_len:.1f}")

    # Optional: detailed comparison of first few frames
    print("\n\n=== First 3 frames detail ===")
    for i in range(min(3, len(autoware_logs), len(cuda_logs))):
        compare_alignments(autoware_logs[i], cuda_logs[i], i)

    return 0


if __name__ == '__main__':
    sys.exit(main())
