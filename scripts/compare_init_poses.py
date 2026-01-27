#!/usr/bin/env python3
"""
Compare initial poses from CUDA and Autoware rosbag recordings.

Extracts the first pose from /localization/pose_estimator/initial_pose_with_covariance
topic in each rosbag and computes the difference.

Usage:
    python3 scripts/compare_init_poses.py <cuda_rosbag> <autoware_rosbag>
    python3 scripts/compare_init_poses.py rosbag/cuda_20260128_050012 rosbag/builtin_20260128_050155
"""

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Optional, Tuple

# ROS2 imports
try:
    from rclpy.serialization import deserialize_message
    from geometry_msgs.msg import PoseWithCovarianceStamped
except ImportError:
    print("Error: ROS2 Python packages not found. Source your ROS2 environment first.")
    sys.exit(1)


def find_db3_file(bag_path: Path) -> Optional[Path]:
    """Find the .db3 file in a rosbag directory."""
    bag_path = Path(bag_path)
    if bag_path.is_file() and bag_path.suffix == '.db3':
        return bag_path

    # Look for db3 files in the directory
    if bag_path.is_dir():
        db3_files = list(bag_path.glob("*.db3"))
        if db3_files:
            return sorted(db3_files)[0]  # Return first one (usually _0.db3)

    return None


def read_first_pose(bag_path: str, topic: str) -> Optional[PoseWithCovarianceStamped]:
    """Read first message from a topic in rosbag."""
    db_path = find_db3_file(Path(bag_path))
    if not db_path:
        print(f"Error: No .db3 file found in {bag_path}")
        return None

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get topic ID
        cursor.execute("SELECT id FROM topics WHERE name=?", (topic,))
        row = cursor.fetchone()
        if not row:
            print(f"Warning: Topic {topic} not found in {db_path}")
            conn.close()
            return None
        topic_id = row[0]

        # Get first message
        cursor.execute(
            "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1",
            (topic_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            print(f"Warning: No messages found for topic {topic}")
            return None

        # Deserialize
        msg = deserialize_message(row[0], PoseWithCovarianceStamped)
        return msg
    except Exception as e:
        print(f"Error reading {bag_path}: {e}")
        return None


def quat_to_rpy(q) -> Tuple[float, float, float]:
    """Extract roll, pitch, yaw from quaternion."""
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (q.w * q.y - q.z * q.x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def pose_to_array(msg: PoseWithCovarianceStamped) -> list:
    """Convert PoseWithCovarianceStamped to [x, y, z, roll, pitch, yaw]."""
    p = msg.pose.pose.position
    roll, pitch, yaw = quat_to_rpy(msg.pose.pose.orientation)
    return [p.x, p.y, p.z, roll, pitch, yaw]


def compare_poses(cuda_pose: list, aw_pose: list) -> dict:
    """Compare two poses and return differences."""
    # Translation difference (Euclidean)
    dx = cuda_pose[0] - aw_pose[0]
    dy = cuda_pose[1] - aw_pose[1]
    dz = cuda_pose[2] - aw_pose[2]
    trans_diff = math.sqrt(dx*dx + dy*dy + dz*dz)

    # Yaw difference (handle wraparound)
    cuda_yaw = cuda_pose[5]
    aw_yaw = aw_pose[5]
    yaw_diff = abs(cuda_yaw - aw_yaw)
    while yaw_diff > math.pi:
        yaw_diff = 2 * math.pi - yaw_diff

    return {
        'position_diff_m': trans_diff,
        'yaw_diff_deg': math.degrees(yaw_diff),
        'dx': dx,
        'dy': dy,
        'dz': dz,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare initial poses from CUDA and Autoware rosbags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("cuda_rosbag", help="Path to CUDA rosbag directory")
    parser.add_argument("autoware_rosbag", help="Path to Autoware rosbag directory")
    parser.add_argument("--topic", default="/localization/pose_estimator/initial_pose_with_covariance",
                        help="Topic to extract pose from")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--output", "-o", type=Path, help="Write results to file")
    args = parser.parse_args()

    # Read poses
    cuda_msg = read_first_pose(args.cuda_rosbag, args.topic)
    aw_msg = read_first_pose(args.autoware_rosbag, args.topic)

    if not cuda_msg:
        print(f"Error: Could not read CUDA pose from {args.cuda_rosbag}")
        sys.exit(1)
    if not aw_msg:
        print(f"Error: Could not read Autoware pose from {args.autoware_rosbag}")
        sys.exit(1)

    cuda_pose = pose_to_array(cuda_msg)
    aw_pose = pose_to_array(aw_msg)
    diff = compare_poses(cuda_pose, aw_pose)

    result = {
        'cuda_pose': {
            'x': cuda_pose[0],
            'y': cuda_pose[1],
            'z': cuda_pose[2],
            'roll_deg': math.degrees(cuda_pose[3]),
            'pitch_deg': math.degrees(cuda_pose[4]),
            'yaw_deg': math.degrees(cuda_pose[5]),
        },
        'autoware_pose': {
            'x': aw_pose[0],
            'y': aw_pose[1],
            'z': aw_pose[2],
            'roll_deg': math.degrees(aw_pose[3]),
            'pitch_deg': math.degrees(aw_pose[4]),
            'yaw_deg': math.degrees(aw_pose[5]),
        },
        'difference': {
            'position_m': diff['position_diff_m'],
            'yaw_deg': diff['yaw_diff_deg'],
        },
    }

    if args.json:
        output = json.dumps(result, indent=2)
    else:
        output = []
        output.append("=" * 60)
        output.append(" Initial Pose Comparison (from rosbag)")
        output.append("=" * 60)
        output.append("")
        output.append(f"CUDA rosbag:     {args.cuda_rosbag}")
        output.append(f"Autoware rosbag: {args.autoware_rosbag}")
        output.append("")
        output.append("CUDA pose:")
        output.append(f"  x={cuda_pose[0]:.2f}, y={cuda_pose[1]:.2f}, z={cuda_pose[2]:.2f}")
        output.append(f"  yaw={math.degrees(cuda_pose[5]):.1f}°")
        output.append("")
        output.append("Autoware pose:")
        output.append(f"  x={aw_pose[0]:.2f}, y={aw_pose[1]:.2f}, z={aw_pose[2]:.2f}")
        output.append(f"  yaw={math.degrees(aw_pose[5]):.1f}°")
        output.append("")
        output.append("Difference:")
        output.append(f"  Position: {diff['position_diff_m']:.3f} m")
        output.append(f"  Yaw:      {diff['yaw_diff_deg']:.2f}°")
        output.append("")

        # Assessment
        if diff['position_diff_m'] < 0.1 and diff['yaw_diff_deg'] < 1.0:
            output.append("Assessment: PASS - Poses are functionally identical")
        elif diff['position_diff_m'] < 0.5 and diff['yaw_diff_deg'] < 5.0:
            output.append("Assessment: ACCEPTABLE - Poses are within tolerance")
        else:
            output.append("Assessment: FAIL - Poses differ significantly")

        output = "\n".join(output)

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
            if not args.json:
                f.write("\n")
        print(f"Results written to: {args.output}")
    else:
        print(output)

    # Exit with error if poses differ too much
    if diff['position_diff_m'] > 0.5 or diff['yaw_diff_deg'] > 5.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
