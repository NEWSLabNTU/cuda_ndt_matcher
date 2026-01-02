"""Pytest configuration and fixtures for CUDA NDT integration tests."""

import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional
import pytest

# Project paths
PROJECT_DIR = Path(__file__).parent.parent
FIXTURES_DIR = Path(__file__).parent / "fixtures"
GROUND_TRUTH_DIR = FIXTURES_DIR / "ground_truth"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (runs just run-cuda)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


@pytest.fixture(scope="session")
def project_dir() -> Path:
    """Return project root directory."""
    return PROJECT_DIR


@pytest.fixture(scope="session")
def ground_truth_dir() -> Path:
    """Return ground truth fixtures directory."""
    return GROUND_TRUTH_DIR


@pytest.fixture(scope="session")
def ground_truth(ground_truth_dir):
    """
    Load pre-recorded Autoware builtin NDT output as ground truth.

    This fixture requires running ./scripts/collect_ground_truth.sh first.

    Returns:
        dict with 'rosbag', 'debug_log', and 'metadata' paths
    """
    rosbag_path = ground_truth_dir / "rosbag"
    debug_log_path = ground_truth_dir / "debug.jsonl"
    metadata_path = ground_truth_dir / "metadata.json"

    if not rosbag_path.exists():
        pytest.skip(
            f"Ground truth not found at {ground_truth_dir}. "
            "Run ./scripts/collect_ground_truth.sh first."
        )

    metadata = None
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    return {
        "rosbag": rosbag_path,
        "debug_log": debug_log_path if debug_log_path.exists() else None,
        "metadata": metadata,
    }


@pytest.fixture(scope="session")
def cuda_result(project_dir, tmp_path_factory):
    """
    Run CUDA NDT demo and return results.

    This fixture:
    1. Runs `just run-cuda` in the project directory
    2. Collects the resulting rosbag and debug log
    3. Returns paths and metadata

    Note: This is slow (~2 minutes) and only runs once per test session.

    Returns:
        dict with 'rosbag', 'debug_log', 'returncode', 'stdout', 'stderr'
    """
    # Run CUDA demo
    result = subprocess.run(
        ["just", "run-cuda"],
        cwd=project_dir,
        capture_output=True,
        timeout=300,  # 5 minute timeout
    )

    # Find latest CUDA rosbag
    rosbag_dir = project_dir / "rosbag"
    cuda_bags = sorted(rosbag_dir.glob("cuda_*"))
    latest_bag = cuda_bags[-1] if cuda_bags else None

    # Debug log location
    debug_log = Path("/tmp/ndt_cuda_debug.jsonl")

    return {
        "rosbag": latest_bag,
        "debug_log": debug_log if debug_log.exists() else None,
        "returncode": result.returncode,
        "stdout": result.stdout.decode(errors="replace"),
        "stderr": result.stderr.decode(errors="replace"),
    }


@pytest.fixture
def sample_rosbag(project_dir) -> Optional[Path]:
    """Return path to sample input rosbag if available."""
    sample_path = project_dir / "data" / "sample-rosbag-fixed"
    if sample_path.exists():
        return sample_path
    return None


@pytest.fixture
def sample_map(project_dir) -> Optional[Path]:
    """Return path to sample map if available."""
    map_path = project_dir / "data" / "sample-map-rosbag"
    if map_path.exists():
        return map_path
    return None


# Helper functions available to tests

def has_ground_truth() -> bool:
    """Check if ground truth data is available."""
    return (GROUND_TRUTH_DIR / "rosbag").exists()


def has_sample_data() -> bool:
    """Check if sample rosbag and map are available."""
    return (
        (PROJECT_DIR / "data" / "sample-rosbag-fixed").exists() and
        (PROJECT_DIR / "data" / "sample-map-rosbag").exists()
    )
