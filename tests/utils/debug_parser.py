"""Utilities for parsing NDT debug log files."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, Any
import numpy as np


@dataclass
class DebugEntry:
    """A single entry from NDT debug log."""
    timestamp_ns: int
    iterations: int
    converged: bool
    score: float
    nvtl: float
    transform_probability: float
    initial_pose: Optional[dict]
    result_pose: Optional[dict]
    execution_time_ms: float
    raw: dict  # Original JSON entry

    @property
    def timestamp_sec(self) -> float:
        return self.timestamp_ns / 1e9


def parse_debug_log(log_path: Union[str, Path]) -> List[DebugEntry]:
    """
    Parse NDT debug JSONL log file.

    The CUDA NDT node writes debug output to /tmp/ndt_cuda_debug.jsonl
    when NDT_DEBUG=1 is set.

    Args:
        log_path: Path to JSONL debug log

    Returns:
        List of DebugEntry objects sorted by timestamp
    """
    log_path = Path(log_path)

    if not log_path.exists():
        return []

    entries = []
    with open(log_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                entry = _parse_entry(data)
                if entry:
                    entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue

    # Sort by timestamp
    entries.sort(key=lambda e: e.timestamp_ns)
    return entries


def _parse_entry(data: dict) -> Optional[DebugEntry]:
    """Parse a single JSON entry into DebugEntry."""
    try:
        # convergence_status can be "Converged", "MaxIterations", "Oscillation", etc.
        convergence_status = data.get("convergence_status", "")
        converged = convergence_status == "Converged"

        return DebugEntry(
            timestamp_ns=data.get("timestamp_ns", 0),
            iterations=data.get("total_iterations", 0),  # Use total_iterations, not iterations array
            converged=converged,
            score=data.get("final_score", 0.0),
            nvtl=data.get("final_nvtl", 0.0),
            transform_probability=data.get("transform_probability", 0.0),
            initial_pose=data.get("initial_pose"),
            result_pose=data.get("final_pose"),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            raw=data,
        )
    except (KeyError, TypeError):
        return None


def get_iterations(entries: List[DebugEntry]) -> np.ndarray:
    """Extract iteration counts from debug entries."""
    return np.array([e.iterations for e in entries])


def get_execution_times(entries: List[DebugEntry]) -> np.ndarray:
    """Extract execution times from debug entries."""
    return np.array([e.execution_time_ms for e in entries])


def get_scores(entries: List[DebugEntry]) -> np.ndarray:
    """Extract NDT scores from debug entries."""
    return np.array([e.score for e in entries])


def get_nvtl_scores(entries: List[DebugEntry]) -> np.ndarray:
    """Extract NVTL scores from debug entries."""
    return np.array([e.nvtl for e in entries])


def get_convergence_rate(entries: List[DebugEntry]) -> float:
    """Calculate percentage of alignments that converged."""
    if not entries:
        return 0.0
    converged = sum(1 for e in entries if e.converged)
    return converged / len(entries) * 100.0


def debug_statistics(entries: List[DebugEntry]) -> dict:
    """
    Compute statistics from debug entries.

    Args:
        entries: List of debug entries

    Returns:
        Dictionary with various statistics
    """
    if not entries:
        return {
            "count": 0,
            "iterations": {},
            "execution_time_ms": {},
            "nvtl": {},
            "convergence_rate": 0.0,
        }

    iterations = get_iterations(entries)
    exec_times = get_execution_times(entries)
    nvtl_scores = get_nvtl_scores(entries)

    def array_stats(arr: np.ndarray) -> dict:
        if len(arr) == 0:
            return {"min": None, "max": None, "mean": None, "std": None}
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    return {
        "count": len(entries),
        "iterations": array_stats(iterations),
        "execution_time_ms": array_stats(exec_times),
        "nvtl": array_stats(nvtl_scores),
        "convergence_rate": get_convergence_rate(entries),
    }
