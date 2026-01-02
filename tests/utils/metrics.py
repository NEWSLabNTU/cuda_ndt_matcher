"""Metrics for comparing NDT outputs."""

from typing import List, Tuple, Optional
import numpy as np

from .rosbag_parser import TimestampedPose, TimestampedScalar


def align_trajectories_by_time(
    reference: List[TimestampedPose],
    test: List[TimestampedPose],
    tolerance_ms: float = 50.0,
) -> List[Tuple[TimestampedPose, TimestampedPose]]:
    """
    Align two trajectories by timestamp.

    For each test pose, find the closest reference pose within tolerance.

    Args:
        reference: Reference trajectory (ground truth)
        test: Test trajectory to compare
        tolerance_ms: Maximum time difference in milliseconds

    Returns:
        List of (reference_pose, test_pose) pairs
    """
    if not reference or not test:
        return []

    tolerance_ns = tolerance_ms * 1e6
    pairs = []

    # Build sorted list of reference timestamps for binary search
    ref_times = np.array([p.timestamp_ns for p in reference])

    for test_pose in test:
        # Find closest reference timestamp
        idx = np.searchsorted(ref_times, test_pose.timestamp_ns)

        # Check both neighbors
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(reference):
            candidates.append(idx)

        best_idx = None
        best_diff = float('inf')
        for i in candidates:
            diff = abs(reference[i].timestamp_ns - test_pose.timestamp_ns)
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        if best_idx is not None and best_diff <= tolerance_ns:
            pairs.append((reference[best_idx], test_pose))

    return pairs


def trajectory_rmse(
    reference: List[TimestampedPose],
    test: List[TimestampedPose],
    use_2d: bool = True,
) -> float:
    """
    Compute RMSE between two trajectories.

    Args:
        reference: Reference trajectory (ground truth)
        test: Test trajectory
        use_2d: If True, compute 2D RMSE (x, y only). Otherwise 3D.

    Returns:
        RMSE in meters. Returns inf if no aligned pairs found.
    """
    pairs = align_trajectories_by_time(reference, test)

    if not pairs:
        return float('inf')

    errors = []
    for ref, tst in pairs:
        if use_2d:
            diff = ref.position_2d - tst.position_2d
        else:
            diff = ref.position - tst.position
        errors.append(np.linalg.norm(diff))

    return np.sqrt(np.mean(np.array(errors) ** 2))


def trajectory_max_error(
    reference: List[TimestampedPose],
    test: List[TimestampedPose],
    use_2d: bool = True,
) -> float:
    """
    Compute maximum error between two trajectories.

    Args:
        reference: Reference trajectory (ground truth)
        test: Test trajectory
        use_2d: If True, compute 2D error. Otherwise 3D.

    Returns:
        Maximum error in meters. Returns inf if no aligned pairs found.
    """
    pairs = align_trajectories_by_time(reference, test)

    if not pairs:
        return float('inf')

    max_error = 0.0
    for ref, tst in pairs:
        if use_2d:
            diff = ref.position_2d - tst.position_2d
        else:
            diff = ref.position - tst.position
        error = np.linalg.norm(diff)
        max_error = max(max_error, error)

    return max_error


def max_deviation(poses: List[TimestampedPose], use_2d: bool = True) -> float:
    """
    Find maximum jump between consecutive poses.

    This detects divergence or sudden localization jumps.

    Args:
        poses: List of poses sorted by timestamp
        use_2d: If True, compute 2D distance. Otherwise 3D.

    Returns:
        Maximum distance between consecutive poses in meters.
    """
    if len(poses) < 2:
        return 0.0

    max_jump = 0.0
    for i in range(1, len(poses)):
        if use_2d:
            diff = poses[i].position_2d - poses[i - 1].position_2d
        else:
            diff = poses[i].position - poses[i - 1].position
        jump = np.linalg.norm(diff)
        max_jump = max(max_jump, jump)

    return max_jump


def score_statistics(
    scores: List[TimestampedScalar],
) -> dict:
    """
    Compute statistics for a list of scores.

    Args:
        scores: List of timestamped scalar values

    Returns:
        Dictionary with min, max, mean, std, median
    """
    if not scores:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "median": None,
            "count": 0,
        }

    values = np.array([s.value for s in scores])
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "count": len(values),
    }


def compare_score_distributions(
    reference: List[TimestampedScalar],
    test: List[TimestampedScalar],
) -> dict:
    """
    Compare two score distributions.

    Args:
        reference: Reference scores (ground truth)
        test: Test scores

    Returns:
        Dictionary with comparison metrics
    """
    ref_stats = score_statistics(reference)
    test_stats = score_statistics(test)

    if ref_stats["mean"] is None or test_stats["mean"] is None:
        return {
            "reference": ref_stats,
            "test": test_stats,
            "mean_diff": None,
            "mean_ratio": None,
        }

    return {
        "reference": ref_stats,
        "test": test_stats,
        "mean_diff": test_stats["mean"] - ref_stats["mean"],
        "mean_ratio": test_stats["mean"] / ref_stats["mean"] if ref_stats["mean"] != 0 else None,
    }
