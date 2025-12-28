#!/usr/bin/env python3
"""
Compare NDT alignment results between Autoware (builtin) and CUDA implementations.

This script reads:
1. Rosbag topics from both implementations (pose, NVTL, iteration count, etc.)
2. CUDA debug JSON file (when NDT_DEBUG=1 was enabled)

Usage:
    python3 scripts/compare_ndt.py rosbag_builtin/ rosbag_cuda/ [--cuda-debug /tmp/ndt_cuda_debug.jsonl]
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import get_typestore, Stores
except ImportError:
    print("Error: rosbags package not found. Install with: pip install rosbags")
    sys.exit(1)

# Get ROS2 Humble typestore for message deserialization
typestore = get_typestore(Stores.ROS2_HUMBLE)


@dataclass
class PoseData:
    """Pose data from rosbag."""
    timestamp_ns: int
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float


@dataclass
class DebugMetrics:
    """Debug metrics from rosbag."""
    timestamp_ns: int
    transform_probability: Optional[float] = None
    nvtl: Optional[float] = None
    iteration_num: Optional[int] = None
    exe_time_ms: Optional[float] = None


@dataclass
class CudaIterationDebug:
    """Debug info for a single CUDA NDT iteration."""
    iteration: int
    pose: List[float]
    score: float
    step_length: float
    num_correspondences: int
    direction_reversed: bool


@dataclass
class CudaAlignmentDebug:
    """Full debug info for one CUDA NDT alignment."""
    timestamp_ns: int
    initial_pose: List[float]
    final_pose: List[float]
    num_source_points: int
    convergence_status: str
    total_iterations: int
    final_score: float
    final_nvtl: float
    iterations: List[CudaIterationDebug]


def read_rosbag_poses(bag_path: Path, topic: str = "/localization/pose_estimator/pose") -> List[PoseData]:
    """Read pose messages from rosbag."""
    poses = []
    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            print(f"Warning: Topic {topic} not found in {bag_path}")
            return poses

        for connection, timestamp, rawdata in reader.messages(connections):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            poses.append(PoseData(
                timestamp_ns=timestamp,
                x=msg.pose.position.x,
                y=msg.pose.position.y,
                z=msg.pose.position.z,
                qx=msg.pose.orientation.x,
                qy=msg.pose.orientation.y,
                qz=msg.pose.orientation.z,
                qw=msg.pose.orientation.w,
            ))
    return poses


def read_rosbag_debug_metrics(bag_path: Path) -> List[DebugMetrics]:
    """Read debug metrics from rosbag."""
    metrics_by_ts: Dict[int, DebugMetrics] = {}

    topic_map = {
        "/localization/pose_estimator/transform_probability": "transform_probability",
        "/localization/pose_estimator/nearest_voxel_transformation_likelihood": "nvtl",
        "/localization/pose_estimator/iteration_num": "iteration_num",
        "/localization/pose_estimator/exe_time_ms": "exe_time_ms",
    }

    # Try to register tier4_debug_msgs types if not present
    try:
        # Define the custom message types for Autoware
        # Float32Stamped: stamp (builtin_interfaces/Time) + data (float32)
        # Int32Stamped: stamp (builtin_interfaces/Time) + data (int32)
        from rosbags.typesys import register_types
        from rosbags.typesys.types import FIELDDEFS

        if 'tier4_debug_msgs/msg/Float32Stamped' not in typestore.fielddefs:
            # Simple approach: just skip these topics
            print("  Note: tier4_debug_msgs not registered, skipping debug metrics")
            return []
    except Exception:
        pass

    with Reader(bag_path) as reader:
        for topic, field in topic_map.items():
            connections = [c for c in reader.connections if c.topic == topic]
            if not connections:
                continue

            try:
                for connection, timestamp, rawdata in reader.messages(connections):
                    try:
                        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

                        # Use message timestamp, not bag timestamp
                        msg_ts = msg.stamp.sec * 1_000_000_000 + msg.stamp.nanosec

                        if msg_ts not in metrics_by_ts:
                            metrics_by_ts[msg_ts] = DebugMetrics(timestamp_ns=msg_ts)

                        setattr(metrics_by_ts[msg_ts], field, msg.data)
                    except KeyError:
                        # Message type not registered, skip
                        break
            except Exception as e:
                print(f"  Warning: Could not read topic {topic}: {e}")

    return sorted(metrics_by_ts.values(), key=lambda m: m.timestamp_ns)


def read_cuda_debug_json(json_path: Path) -> List[CudaAlignmentDebug]:
    """Read CUDA debug JSON lines file."""
    alignments = []

    if not json_path.exists():
        print(f"Warning: CUDA debug file not found: {json_path}")
        return alignments

    with open(json_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                iterations = [
                    CudaIterationDebug(
                        iteration=it["iteration"],
                        pose=it["pose"],
                        score=it["score"],
                        step_length=it["step_length"],
                        num_correspondences=it["num_correspondences"],
                        direction_reversed=it["direction_reversed"],
                    )
                    for it in data.get("iterations", [])
                ]
                alignments.append(CudaAlignmentDebug(
                    timestamp_ns=data["timestamp_ns"],
                    initial_pose=data["initial_pose"],
                    final_pose=data["final_pose"],
                    num_source_points=data["num_source_points"],
                    convergence_status=data["convergence_status"],
                    total_iterations=data["total_iterations"],
                    final_score=data["final_score"],
                    final_nvtl=data["final_nvtl"],
                    iterations=iterations,
                ))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to parse JSON line: {e}")

    return alignments


def pose_distance(p1: PoseData, p2: PoseData) -> float:
    """Calculate Euclidean distance between two poses."""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)


def find_nearest_by_timestamp(target_ts: int, candidates: list, tolerance_ns: int = 50_000_000) -> Optional[Any]:
    """Find the candidate with the closest timestamp within tolerance."""
    best = None
    best_diff = tolerance_ns

    for c in candidates:
        diff = abs(c.timestamp_ns - target_ts)
        if diff < best_diff:
            best_diff = diff
            best = c

    return best


def compare_implementations(
    builtin_path: Path,
    cuda_path: Path,
    cuda_debug_path: Optional[Path] = None,
):
    """Compare builtin and CUDA NDT implementations."""

    print("=" * 70)
    print("NDT Implementation Comparison")
    print("=" * 70)
    print(f"Builtin rosbag: {builtin_path}")
    print(f"CUDA rosbag: {cuda_path}")
    if cuda_debug_path:
        print(f"CUDA debug JSON: {cuda_debug_path}")
    print()

    # Read poses
    print("Reading poses...")
    builtin_poses = read_rosbag_poses(builtin_path)
    cuda_poses = read_rosbag_poses(cuda_path)
    print(f"  Builtin: {len(builtin_poses)} poses")
    print(f"  CUDA: {len(cuda_poses)} poses")
    print()

    # Read debug metrics
    print("Reading debug metrics...")
    builtin_metrics = read_rosbag_debug_metrics(builtin_path)
    cuda_metrics = read_rosbag_debug_metrics(cuda_path)
    print(f"  Builtin: {len(builtin_metrics)} entries")
    print(f"  CUDA: {len(cuda_metrics)} entries")
    print()

    # Read CUDA debug JSON if available
    cuda_debug = []
    if cuda_debug_path:
        print("Reading CUDA debug JSON...")
        cuda_debug = read_cuda_debug_json(cuda_debug_path)
        print(f"  {len(cuda_debug)} alignment records")
        print()

    # Compare poses at matching timestamps
    print("-" * 70)
    print("POSE COMPARISON")
    print("-" * 70)

    pose_diffs = []
    for bp in builtin_poses:
        cp = find_nearest_by_timestamp(bp.timestamp_ns, cuda_poses)
        if cp:
            dist = pose_distance(bp, cp)
            pose_diffs.append(dist)

    if pose_diffs:
        print(f"Matched poses: {len(pose_diffs)}")
        print(f"Position difference (m):")
        print(f"  Mean:   {np.mean(pose_diffs):.4f}")
        print(f"  Std:    {np.std(pose_diffs):.4f}")
        print(f"  Min:    {np.min(pose_diffs):.4f}")
        print(f"  Max:    {np.max(pose_diffs):.4f}")
        print(f"  Median: {np.median(pose_diffs):.4f}")
    else:
        print("No matching poses found!")
    print()

    # Compare debug metrics
    print("-" * 70)
    print("DEBUG METRICS COMPARISON")
    print("-" * 70)

    # Iteration counts
    builtin_iters = [m.iteration_num for m in builtin_metrics if m.iteration_num is not None]
    cuda_iters = [m.iteration_num for m in cuda_metrics if m.iteration_num is not None]

    if builtin_iters and cuda_iters:
        print(f"Iteration counts:")
        print(f"  Builtin: mean={np.mean(builtin_iters):.1f}, max={max(builtin_iters)}")
        print(f"  CUDA:    mean={np.mean(cuda_iters):.1f}, max={max(cuda_iters)}")

    # NVTL scores
    builtin_nvtl = [m.nvtl for m in builtin_metrics if m.nvtl is not None]
    cuda_nvtl = [m.nvtl for m in cuda_metrics if m.nvtl is not None]

    if builtin_nvtl and cuda_nvtl:
        print(f"\nNVTL scores:")
        print(f"  Builtin: mean={np.mean(builtin_nvtl):.4f}, std={np.std(builtin_nvtl):.4f}")
        print(f"  CUDA:    mean={np.mean(cuda_nvtl):.4f}, std={np.std(cuda_nvtl):.4f}")

    # Transform probability
    builtin_tp = [m.transform_probability for m in builtin_metrics if m.transform_probability is not None]
    cuda_tp = [m.transform_probability for m in cuda_metrics if m.transform_probability is not None]

    if builtin_tp and cuda_tp:
        print(f"\nTransform probability:")
        print(f"  Builtin: mean={np.mean(builtin_tp):.4f}, std={np.std(builtin_tp):.4f}")
        print(f"  CUDA:    mean={np.mean(cuda_tp):.4f}, std={np.std(cuda_tp):.4f}")

    # Execution time
    builtin_exe = [m.exe_time_ms for m in builtin_metrics if m.exe_time_ms is not None]
    cuda_exe = [m.exe_time_ms for m in cuda_metrics if m.exe_time_ms is not None]

    if builtin_exe and cuda_exe:
        print(f"\nExecution time (ms):")
        print(f"  Builtin: mean={np.mean(builtin_exe):.1f}, max={max(builtin_exe):.1f}")
        print(f"  CUDA:    mean={np.mean(cuda_exe):.1f}, max={max(cuda_exe):.1f}")
    print()

    # CUDA iteration details
    if cuda_debug:
        print("-" * 70)
        print("CUDA ITERATION DETAILS (first 5 alignments)")
        print("-" * 70)

        for i, align in enumerate(cuda_debug[:5]):
            print(f"\nAlignment {i + 1}: ts={align.timestamp_ns}")
            print(f"  Status: {align.convergence_status}, iterations: {align.total_iterations}")
            print(f"  Points: {align.num_source_points}, score: {align.final_score:.4f}, nvtl: {align.final_nvtl:.4f}")
            print(f"  Initial: [{', '.join(f'{x:.4f}' for x in align.initial_pose)}]")
            print(f"  Final:   [{', '.join(f'{x:.4f}' for x in align.final_pose)}]")

            if align.iterations:
                print("  Iterations:")
                for it in align.iterations[:10]:  # Show first 10 iterations
                    print(f"    {it.iteration}: score={it.score:.4f}, step={it.step_length:.6f}, "
                          f"corr={it.num_correspondences}, rev={it.direction_reversed}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    issues = []
    if pose_diffs and np.mean(pose_diffs) > 0.1:
        issues.append(f"Large pose difference: mean={np.mean(pose_diffs):.3f}m")
    if cuda_iters and np.mean(cuda_iters) > 10:
        issues.append(f"High CUDA iteration count: mean={np.mean(cuda_iters):.1f}")
    if cuda_nvtl and np.mean(cuda_nvtl) < 1.5:
        issues.append(f"Low CUDA NVTL: mean={np.mean(cuda_nvtl):.3f}")

    if issues:
        print("Potential issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No major issues detected.")


def main():
    parser = argparse.ArgumentParser(description="Compare NDT implementations")
    parser.add_argument("builtin_bag", type=Path, help="Path to builtin rosbag directory")
    parser.add_argument("cuda_bag", type=Path, help="Path to CUDA rosbag directory")
    parser.add_argument("--cuda-debug", type=Path, default=Path("/tmp/ndt_cuda_debug.jsonl"),
                        help="Path to CUDA debug JSON file")

    args = parser.parse_args()

    if not args.builtin_bag.exists():
        print(f"Error: Builtin rosbag not found: {args.builtin_bag}")
        sys.exit(1)

    if not args.cuda_bag.exists():
        print(f"Error: CUDA rosbag not found: {args.cuda_bag}")
        sys.exit(1)

    compare_implementations(
        args.builtin_bag,
        args.cuda_bag,
        args.cuda_debug if args.cuda_debug.exists() else None,
    )


if __name__ == "__main__":
    main()
