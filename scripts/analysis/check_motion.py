#!/usr/bin/env python3
"""Check if vehicle is stationary or moving in early frames."""

import sys
from pathlib import Path

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import get_typestore, Stores
except ImportError:
    sys.exit(1)

typestore = get_typestore(Stores.ROS2_HUMBLE)


def read_poses(bag_path: Path, topic: str):
    poses = []
    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            return poses
        for connection, timestamp, rawdata in reader.messages(connections):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            poses.append({
                'x': msg.pose.position.x,
                'y': msg.pose.position.y,
            })
    return poses


builtin_poses = read_poses(Path("rosbag/builtin_20251228_185253"), "/localization/pose_estimator/pose")
cuda_poses = read_poses(Path("rosbag/cuda_20251228_185403"), "/localization/pose_estimator/pose")

print("=== Builtin trajectory (first 70 frames) ===")
print("Frame | X          | Y          | ΔX     | ΔY")
print("-" * 55)
for i in range(min(70, len(builtin_poses))):
    p = builtin_poses[i]
    if i == 0:
        dx, dy = 0, 0
    else:
        dx = p['x'] - builtin_poses[i-1]['x']
        dy = p['y'] - builtin_poses[i-1]['y']

    if i % 5 == 0 or abs(dx) > 0.1 or abs(dy) > 0.1:
        print(f"{i:5d} | {p['x']:10.4f} | {p['y']:10.4f} | {dx:+6.3f} | {dy:+6.3f}")

print("\n=== CUDA trajectory (first 70 frames) ===")
print("Frame | X          | Y          | ΔX     | ΔY")
print("-" * 55)
for i in range(min(70, len(cuda_poses))):
    p = cuda_poses[i]
    if i == 0:
        dx, dy = 0, 0
    else:
        dx = p['x'] - cuda_poses[i-1]['x']
        dy = p['y'] - cuda_poses[i-1]['y']

    if i % 5 == 0 or abs(dx) > 0.1 or abs(dy) > 0.1:
        print(f"{i:5d} | {p['x']:10.4f} | {p['y']:10.4f} | {dx:+6.3f} | {dy:+6.3f}")
