#!/usr/bin/env python3
"""Analyze early frames where divergence starts."""

import json

with open('/tmp/ndt_cuda_debug.jsonl') as f:
    alignments = [json.loads(line) for line in f if line.strip()]

print("=== CUDA NDT behavior at divergence onset (frames 45-65) ===")
print()

for idx in range(45, min(65, len(alignments))):
    a = alignments[idx]
    status = "MAX" if a['convergence_status'] == 'MaxIterations' else "OK "
    corr = a['iterations'][-1]['num_correspondences'] if a['iterations'] else 0

    # Check for issues
    issues = []
    if a['total_iterations'] > 15:
        issues.append("high_iter")
    if a['final_nvtl'] < 2.5:
        issues.append("low_nvtl")
    if corr < 7000:
        issues.append("low_corr")

    # Compute pose correction magnitude
    init = a['initial_pose']
    final = a['final_pose']
    delta_pos = ((final[0]-init[0])**2 + (final[1]-init[1])**2)**0.5

    issue_str = f" [{', '.join(issues)}]" if issues else ""

    print(f"Frame {idx:3d}: [{status}] iter={a['total_iterations']:2d}, nvtl={a['final_nvtl']:.3f}, "
          f"corr={corr:5d}, correction={delta_pos:.3f}m{issue_str}")

    # Show iteration detail if high iteration count
    if a['total_iterations'] > 10:
        print("           Iteration scores:", end=" ")
        for it in a['iterations'][:5]:
            print(f"{it['score']:.0f}", end=" ")
        print("...")

print()
print("=== Comparing frame 50 (good) vs 55 (starting to drift) ===")

for idx in [50, 55, 60]:
    if idx >= len(alignments):
        continue
    a = alignments[idx]
    print(f"\n--- Frame {idx} ---")
    print(f"Status: {a['convergence_status']}, iterations: {a['total_iterations']}")
    print(f"NVTL: {a['final_nvtl']:.4f}, Score: {a['final_score']:.1f}")
    print(f"Initial: [{', '.join(f'{x:.4f}' for x in a['initial_pose'][:2])}]")
    print(f"Final:   [{', '.join(f'{x:.4f}' for x in a['final_pose'][:2])}]")

    if a['iterations']:
        first_it = a['iterations'][0]
        last_it = a['iterations'][-1]
        print(f"First iter: score={first_it['score']:.1f}, corr={first_it['num_correspondences']}")
        print(f"Last iter:  score={last_it['score']:.1f}, corr={last_it['num_correspondences']}")
