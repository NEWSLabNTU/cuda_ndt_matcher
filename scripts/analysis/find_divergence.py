#!/usr/bin/env python3
"""Find where CUDA trajectory diverges from builtin."""

import sys
from pathlib import Path

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import get_typestore, Stores
except ImportError:
    print("Error: rosbags package not found")
    sys.exit(1)

typestore = get_typestore(Stores.ROS2_HUMBLE)


def read_poses(bag_path: Path, topic: str):
    """Read poses from rosbag."""
    poses = []
    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
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
    builtin_path = Path("rosbag/builtin_20251228_185253")
    cuda_path = Path("rosbag/cuda_20251228_185403")

    pose_topic = "/localization/pose_estimator/pose"

    builtin_poses = read_poses(builtin_path, pose_topic)
    cuda_poses = read_poses(cuda_path, pose_topic)

    print("=== Finding divergence point ===")
    print("Looking for first frame where drift > 0.5m...")
    print()

    prev_dist = 0
    for i in range(min(len(builtin_poses), len(cuda_poses))):
        bp = builtin_poses[i]
        cp = cuda_poses[i]
        dx = cp['x'] - bp['x']
        dy = cp['y'] - bp['y']
        dist = (dx**2 + dy**2)**0.5

        if dist > 0.5 and prev_dist <= 0.5:
            print(f"*** FIRST DIVERGENCE at frame {i} ***")
            print(f"  Builtin: ({bp['x']:.4f}, {bp['y']:.4f})")
            print(f"  CUDA:    ({cp['x']:.4f}, {cp['y']:.4f})")
            print(f"  Drift:   {dist:.4f}m")
            print()

            # Show context
            print("Context (frames before/after):")
            for j in range(max(0, i-5), min(i+10, len(builtin_poses), len(cuda_poses))):
                b = builtin_poses[j]
                c = cuda_poses[j]
                d = ((c['x'] - b['x'])**2 + (c['y'] - b['y'])**2)**0.5
                marker = " <--" if j == i else ""
                print(f"  Frame {j:3d}: drift={d:.4f}m{marker}")
            break

        prev_dist = dist
    else:
        print("No significant divergence found!")

    # Also print summary of drift progression
    print("\n=== Drift progression summary ===")
    milestones = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for threshold in milestones:
        for i in range(min(len(builtin_poses), len(cuda_poses))):
            bp = builtin_poses[i]
            cp = cuda_poses[i]
            dist = ((cp['x'] - bp['x'])**2 + (cp['y'] - bp['y'])**2)**0.5
            if dist >= threshold:
                print(f"  Drift > {threshold:4.1f}m: first at frame {i}")
                break
        else:
            print(f"  Drift > {threshold:4.1f}m: never reached")


if __name__ == "__main__":
    main()
