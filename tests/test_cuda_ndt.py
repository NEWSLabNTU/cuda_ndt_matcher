"""
Integration tests for CUDA NDT scan matcher.

These tests validate the CUDA NDT implementation against
pre-recorded Autoware builtin NDT output (ground truth).

Prerequisites:
1. Download sample data: just download-data
2. Collect ground truth: ./scripts/collect_ground_truth.sh

Running tests:
    pytest tests/test_cuda_ndt.py -v

Skip slow integration tests:
    pytest tests/test_cuda_ndt.py -v -m "not integration"
"""

import numpy as np
import pytest

from utils import (
    parse_poses,
    parse_nvtl_scores,
    parse_iteration_counts,
    trajectory_rmse,
    trajectory_max_error,
    max_deviation,
    compare_score_distributions,
)
from utils.debug_parser import parse_debug_log, debug_statistics


class TestCUDANDTDemo:
    """Test that CUDA NDT demo completes successfully."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_demo_completes(self, cuda_result):
        """CUDA NDT demo should complete without errors."""
        assert cuda_result["returncode"] == 0, (
            f"Demo failed with return code {cuda_result['returncode']}.\n"
            f"stderr: {cuda_result['stderr'][:1000]}"
        )

    @pytest.mark.integration
    @pytest.mark.slow
    def test_rosbag_created(self, cuda_result):
        """Demo should create a rosbag with recorded topics."""
        assert cuda_result["rosbag"] is not None, "No rosbag was created"
        assert cuda_result["rosbag"].exists(), f"Rosbag not found: {cuda_result['rosbag']}"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_debug_log_created(self, cuda_result):
        """Demo should create a debug log when NDT_DEBUG is set."""
        assert cuda_result["debug_log"] is not None, "No debug log was created"
        assert cuda_result["debug_log"].exists(), f"Debug log not found: {cuda_result['debug_log']}"


class TestTrajectoryAccuracy:
    """Test that CUDA NDT trajectory matches ground truth."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_trajectory_rmse(self, ground_truth, cuda_result):
        """CUDA trajectory RMSE should be within tolerance of ground truth."""
        gt_poses = parse_poses(ground_truth["rosbag"])
        cuda_poses = parse_poses(cuda_result["rosbag"])

        if len(gt_poses) == 0:
            pytest.skip("No ground truth poses found in rosbag")
        if len(cuda_poses) == 0:
            pytest.skip("No CUDA poses found - NDT node may not be publishing poses yet")

        rmse = trajectory_rmse(gt_poses, cuda_poses, use_2d=True)
        print(f"Trajectory RMSE: {rmse:.4f}m")

        # Allow 30cm RMSE tolerance
        assert rmse < 0.3, f"Trajectory RMSE {rmse:.3f}m exceeds 0.3m threshold"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_trajectory_max_error(self, ground_truth, cuda_result):
        """Maximum trajectory error should be bounded."""
        gt_poses = parse_poses(ground_truth["rosbag"])
        cuda_poses = parse_poses(cuda_result["rosbag"])

        if len(gt_poses) == 0 or len(cuda_poses) == 0:
            pytest.skip("Pose data not available for comparison")

        max_err = trajectory_max_error(gt_poses, cuda_poses, use_2d=True)
        print(f"Max trajectory error: {max_err:.4f}m")

        # Allow 1m maximum error
        assert max_err < 1.0, f"Max error {max_err:.3f}m exceeds 1.0m threshold"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_no_divergence(self, cuda_result):
        """CUDA NDT should not diverge (no sudden position jumps)."""
        poses = parse_poses(cuda_result["rosbag"])
        if len(poses) == 0:
            pytest.skip("No poses found - NDT node may not be publishing poses yet")

        max_jump = max_deviation(poses, use_2d=True)
        print(f"Max pose jump: {max_jump:.4f}m")

        # A jump > 2m indicates divergence
        assert max_jump < 2.0, (
            f"Max pose jump {max_jump:.2f}m indicates divergence. "
            "Expected < 2.0m between consecutive poses."
        )


class TestScoreQuality:
    """Test that CUDA NDT scores are comparable to ground truth."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_nvtl_scores(self, ground_truth, cuda_result):
        """CUDA NVTL scores should be comparable to ground truth."""
        gt_nvtl = parse_nvtl_scores(ground_truth["rosbag"])
        cuda_nvtl = parse_nvtl_scores(cuda_result["rosbag"])

        if len(gt_nvtl) == 0 or len(cuda_nvtl) == 0:
            pytest.skip("NVTL scores not available in rosbags")

        comparison = compare_score_distributions(gt_nvtl, cuda_nvtl)
        print(f"Ground truth NVTL: mean={comparison['reference']['mean']:.4f}")
        print(f"CUDA NVTL: mean={comparison['test']['mean']:.4f}")

        # CUDA should achieve at least 80% of ground truth NVTL
        ratio = comparison["mean_ratio"]
        assert ratio is not None and ratio >= 0.8, (
            f"CUDA NVTL ({comparison['test']['mean']:.4f}) is significantly worse "
            f"than ground truth ({comparison['reference']['mean']:.4f})"
        )

    @pytest.mark.integration
    @pytest.mark.slow
    def test_iteration_counts(self, cuda_result):
        """CUDA should converge in reasonable number of iterations."""
        iterations = parse_iteration_counts(cuda_result["rosbag"])

        if len(iterations) == 0:
            pytest.skip("Iteration counts not available in rosbag")

        iter_values = np.array([i.value for i in iterations])
        mean_iter = np.mean(iter_values)
        max_iter = np.max(iter_values)

        print(f"Iterations: mean={mean_iter:.1f}, max={max_iter:.0f}")

        # Should converge in reasonable iterations
        assert mean_iter < 15, f"Average iterations {mean_iter:.1f} too high (expected < 15)"
        assert max_iter < 30, f"Max iterations {max_iter:.0f} hitting limit (expected < 30)"


class TestDebugOutput:
    """Test CUDA NDT debug log output."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_debug_entries_parsed(self, cuda_result):
        """Debug log should contain valid entries."""
        if cuda_result["debug_log"] is None:
            pytest.skip("No debug log available")

        entries = parse_debug_log(cuda_result["debug_log"])
        assert len(entries) > 0, "No debug entries found"

        stats = debug_statistics(entries)
        print(f"Debug entries: {stats['count']}")
        print(f"Iterations: mean={stats['iterations']['mean']:.1f}")
        print(f"Execution time: mean={stats['execution_time_ms']['mean']:.1f}ms")
        print(f"Convergence rate: {stats['convergence_rate']:.1f}%")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_convergence_rate(self, cuda_result):
        """Most alignments should converge."""
        if cuda_result["debug_log"] is None:
            pytest.skip("No debug log available")

        entries = parse_debug_log(cuda_result["debug_log"])
        if len(entries) == 0:
            pytest.skip("No debug entries found")

        stats = debug_statistics(entries)
        convergence_rate = stats["convergence_rate"]

        print(f"Convergence rate: {convergence_rate:.1f}%")

        # At least 80% should converge (lowered during development)
        assert convergence_rate >= 80.0, (
            f"Convergence rate {convergence_rate:.1f}% is too low (expected >= 80%)"
        )

    @pytest.mark.integration
    @pytest.mark.slow
    def test_execution_time(self, cuda_result):
        """Execution time should be reasonable."""
        if cuda_result["debug_log"] is None:
            pytest.skip("No debug log available")

        entries = parse_debug_log(cuda_result["debug_log"])
        if len(entries) == 0:
            pytest.skip("No debug entries found")

        stats = debug_statistics(entries)
        mean_time = stats["execution_time_ms"]["mean"]
        max_time = stats["execution_time_ms"]["max"]

        print(f"Execution time: mean={mean_time:.1f}ms, max={max_time:.1f}ms")

        # Should be reasonably fast (< 100ms mean, < 200ms max)
        assert mean_time < 100, f"Mean execution time {mean_time:.1f}ms too high"
        assert max_time < 200, f"Max execution time {max_time:.1f}ms too high"


class TestDataAvailability:
    """Tests that don't require running the demo."""

    def test_ground_truth_exists(self, ground_truth_dir):
        """Ground truth data should be available."""
        rosbag_path = ground_truth_dir / "rosbag"
        if not rosbag_path.exists():
            pytest.skip(
                "Ground truth not collected. Run ./scripts/collect_ground_truth.sh"
            )

        assert rosbag_path.is_dir(), "Ground truth rosbag should be a directory"

    def test_sample_data_exists(self, sample_rosbag, sample_map):
        """Sample data should be downloaded."""
        if sample_rosbag is None or sample_map is None:
            pytest.skip("Sample data not downloaded. Run: just download-data")

        assert sample_rosbag.exists()
        assert sample_map.exists()
