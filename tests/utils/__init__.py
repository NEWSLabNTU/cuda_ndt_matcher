"""Test utilities for CUDA NDT validation."""

from .rosbag_parser import (
    TimestampedPose,
    parse_poses,
    parse_nvtl_scores,
    parse_iteration_counts,
    parse_transform_probability,
)
from .metrics import (
    trajectory_rmse,
    trajectory_max_error,
    max_deviation,
    align_trajectories_by_time,
    compare_score_distributions,
)
from .debug_parser import (
    parse_debug_log,
    DebugEntry,
    debug_statistics,
)

__all__ = [
    "TimestampedPose",
    "parse_poses",
    "parse_nvtl_scores",
    "parse_iteration_counts",
    "parse_transform_probability",
    "trajectory_rmse",
    "trajectory_max_error",
    "max_deviation",
    "align_trajectories_by_time",
    "compare_score_distributions",
    "parse_debug_log",
    "DebugEntry",
    "debug_statistics",
]
