#!/usr/bin/env python3
"""Check if NDT initial pose equals previous NDT final pose."""

import json

with open('/tmp/ndt_cuda_debug.jsonl') as f:
    alignments = [json.loads(line) for line in f if line.strip()]

print("=== Checking if NDT initial = previous NDT final ===")
print("(This would indicate we're using NDT output as next initial guess)")
print()
print("Frame | Prev Final              | Curr Initial            | Match?")
print("-" * 75)

for idx in range(45, min(65, len(alignments))):
    prev = alignments[idx - 1]
    curr = alignments[idx]

    prev_final_x, prev_final_y = prev['final_pose'][0], prev['final_pose'][1]
    curr_init_x, curr_init_y = curr['initial_pose'][0], curr['initial_pose'][1]

    diff = ((curr_init_x - prev_final_x)**2 + (curr_init_y - prev_final_y)**2)**0.5

    match = "YES" if diff < 0.01 else "NO"
    if diff < 0.01:
        match = "YES ✓"
    elif diff < 0.1:
        match = "close"
    else:
        match = "NO"

    print(f"{idx:5d} | ({prev_final_x:9.4f},{prev_final_y:9.4f}) | ({curr_init_x:9.4f},{curr_init_y:9.4f}) | {diff:.4f}m {match}")
