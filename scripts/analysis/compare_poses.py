#!/usr/bin/env python3
"""Compare poses from builtin and CUDA rosbags."""

import sys
from pathlib import Path

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import get_typestore, Stores
except ImportError:
    print("Error: rosbags package not found. Install with: pip install rosbags")
    sys.exit(1)

typestore = get_typestore(Stores.ROS2_HUMBLE)


def read_poses(bag_path: Path, topic: str):
    """Read poses from rosbag."""
    poses = []
    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            print(f"Warning: Topic {topic} not found in {bag_path}")
            return poses

        for connection, timestamp, rawdata in reader.messages(connections):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            poses.append({
                'timestamp_ns': timestamp,
                'x': msg.pose.position.x,
                'y': msg.pose.position.y,
                'z': msg.pose.position.z,
            })
    return poses


def main():
    builtin_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("rosbag/builtin_20251228_185253")
    cuda_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("rosbag/cuda_20251228_185403")

    pose_topic = "/localization/pose_estimator/pose"

    print("Reading builtin poses...")
    builtin_poses = read_poses(builtin_path, pose_topic)
    print(f"  Found {len(builtin_poses)} poses")

    print("Reading CUDA poses...")
    cuda_poses = read_poses(cuda_path, pose_topic)
    print(f"  Found {len(cuda_poses)} poses")

    if not builtin_poses or not cuda_poses:
        print("Cannot compare - missing data")
        return

    # Show first few poses from each
    print("\n=== First 10 BUILTIN poses ===")
    for i, p in enumerate(builtin_poses[:10]):
        print(f"  {i}: x={p['x']:.4f}, y={p['y']:.4f}, z={p['z']:.4f}")

    print("\n=== First 10 CUDA poses ===")
    for i, p in enumerate(cuda_poses[:10]):
        print(f"  {i}: x={p['x']:.4f}, y={p['y']:.4f}, z={p['z']:.4f}")

    # Show trajectory divergence over time
    print("\n=== Position drift over time ===")
    print("Comparing CUDA pose relative to builtin at same index")

    for i in [0, 10, 50, 100, 150, 200]:
        if i >= len(builtin_poses) or i >= len(cuda_poses):
            break
        bp = builtin_poses[i]
        cp = cuda_poses[i]
        dx = cp['x'] - bp['x']
        dy = cp['y'] - bp['y']
        dist = (dx**2 + dy**2)**0.5
        print(f"  Frame {i:3d}: dx={dx:+8.3f}m, dy={dy:+8.3f}m, dist={dist:.3f}m")

    # Show consecutive frame deltas for CUDA
    print("\n=== CUDA frame-to-frame motion ===")
    print("(Position change between consecutive NDT outputs)")
    for i in range(1, min(11, len(cuda_poses))):
        prev = cuda_poses[i-1]
        curr = cuda_poses[i]
        dx = curr['x'] - prev['x']
        dy = curr['y'] - prev['y']
        dist = (dx**2 + dy**2)**0.5
        print(f"  Frame {i-1}->{i}: dx={dx:+.4f}m, dy={dy:+.4f}m, dist={dist:.4f}m")


if __name__ == "__main__":
    main()
