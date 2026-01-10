#!/usr/bin/env python3
"""
Analyze NDT profiling results and generate comparison report.

Parses NDT_DEBUG JSONL files from each mode and computes statistics.
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import statistics


@dataclass
class AlignmentStats:
    """Statistics for a single alignment."""
    timestamp_ns: int
    exe_time_ms: float
    iterations: int
    converged: bool
    score: float
    num_correspondences: int
    oscillation_count: int = 0


@dataclass
class ModeStats:
    """Aggregated statistics for a mode."""
    mode: str
    count: int = 0
    exe_times_ms: list = field(default_factory=list)
    iterations: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    correspondences: list = field(default_factory=list)
    oscillations: list = field(default_factory=list)
    converged_count: int = 0

    def add(self, alignment: AlignmentStats):
        self.count += 1
        self.exe_times_ms.append(alignment.exe_time_ms)
        self.iterations.append(alignment.iterations)
        self.scores.append(alignment.score)
        self.correspondences.append(alignment.num_correspondences)
        self.oscillations.append(alignment.oscillation_count)
        if alignment.converged:
            self.converged_count += 1

    def summary(self) -> dict:
        """Generate summary statistics."""
        if self.count == 0:
            return {"mode": self.mode, "count": 0}

        return {
            "mode": self.mode,
            "count": self.count,
            "converged_pct": 100.0 * self.converged_count / self.count,
            "exe_time_ms": {
                "mean": statistics.mean(self.exe_times_ms),
                "median": statistics.median(self.exe_times_ms),
                "stdev": statistics.stdev(self.exe_times_ms) if len(self.exe_times_ms) > 1 else 0,
                "min": min(self.exe_times_ms),
                "max": max(self.exe_times_ms),
                "p95": sorted(self.exe_times_ms)[int(0.95 * len(self.exe_times_ms))],
                "p99": sorted(self.exe_times_ms)[int(0.99 * len(self.exe_times_ms))],
            },
            "iterations": {
                "mean": statistics.mean(self.iterations),
                "median": statistics.median(self.iterations),
                "max": max(self.iterations),
            },
            "score": {
                "mean": statistics.mean(self.scores),
                "median": statistics.median(self.scores),
            },
            "correspondences": {
                "mean": statistics.mean(self.correspondences),
            },
            "oscillations": {
                "mean": statistics.mean(self.oscillations),
                "max": max(self.oscillations),
            },
        }


def parse_cuda_debug(path: Path) -> list[AlignmentStats]:
    """Parse CUDA NDT debug JSONL file."""
    alignments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Handle both old and new format
                iterations = data.get("iterations", [])
                num_iterations = len(iterations) if isinstance(iterations, list) else data.get("num_iterations", 0)

                # Calculate exe_time from iterations if not present
                exe_time_ms = data.get("exe_time_ms", 0)
                if exe_time_ms == 0 and isinstance(iterations, list) and len(iterations) > 0:
                    # Sum iteration times if available
                    exe_time_ms = sum(it.get("time_ms", 0) for it in iterations)

                alignments.append(AlignmentStats(
                    timestamp_ns=data.get("timestamp_ns", 0),
                    exe_time_ms=exe_time_ms,
                    iterations=num_iterations,
                    converged=data.get("converged", False) or data.get("convergence_status") == "Converged",
                    score=data.get("final_score", 0) or data.get("score", 0),
                    num_correspondences=data.get("num_correspondences", 0),
                    oscillation_count=data.get("oscillation_count", 0),
                ))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}", file=sys.stderr)
                continue
    return alignments


def parse_autoware_debug(path: Path) -> list[AlignmentStats]:
    """Parse Autoware NDT debug JSONL file."""
    alignments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                alignments.append(AlignmentStats(
                    timestamp_ns=data.get("timestamp_ns", 0),
                    exe_time_ms=data.get("exe_time_ms", 0),
                    iterations=data.get("iterations", 0) or data.get("num_iterations", 0),
                    converged=data.get("converged", False),
                    score=data.get("score", 0) or data.get("transform_probability", 0),
                    num_correspondences=data.get("num_correspondences", 0),
                    oscillation_count=0,  # Autoware doesn't track this
                ))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}", file=sys.stderr)
                continue
    return alignments


def format_summary(stats: dict) -> str:
    """Format summary statistics as text."""
    if stats["count"] == 0:
        return f"{stats['mode']}: No data\n"

    lines = [
        f"=== {stats['mode']} ===",
        f"  Alignments: {stats['count']}",
        f"  Converged: {stats['converged_pct']:.1f}%",
        "",
        "  Execution Time (ms):",
        f"    Mean:   {stats['exe_time_ms']['mean']:.2f}",
        f"    Median: {stats['exe_time_ms']['median']:.2f}",
        f"    Stdev:  {stats['exe_time_ms']['stdev']:.2f}",
        f"    Min:    {stats['exe_time_ms']['min']:.2f}",
        f"    Max:    {stats['exe_time_ms']['max']:.2f}",
        f"    P95:    {stats['exe_time_ms']['p95']:.2f}",
        f"    P99:    {stats['exe_time_ms']['p99']:.2f}",
        "",
        "  Iterations:",
        f"    Mean:   {stats['iterations']['mean']:.1f}",
        f"    Median: {stats['iterations']['median']:.0f}",
        f"    Max:    {stats['iterations']['max']}",
        "",
        f"  Score (mean): {stats['score']['mean']:.2f}",
        f"  Correspondences (mean): {stats['correspondences']['mean']:.0f}",
    ]

    if stats['oscillations']['max'] > 0:
        lines.extend([
            f"  Oscillations (mean/max): {stats['oscillations']['mean']:.1f} / {stats['oscillations']['max']}",
        ])

    return "\n".join(lines) + "\n"


def format_comparison(all_stats: list[dict]) -> str:
    """Format comparison table."""
    if len(all_stats) < 2:
        return ""

    # Find baseline (builtin or first mode)
    baseline = next((s for s in all_stats if s["mode"] == "builtin"), all_stats[0])
    if baseline["count"] == 0:
        return "No baseline data for comparison\n"

    lines = [
        "",
        "=== Comparison vs Baseline ({}) ===".format(baseline["mode"]),
        "",
        "| Mode | Exe Time (ms) | Speedup | Iterations | Converged |",
        "|------|---------------|---------|------------|-----------|",
    ]

    for stats in all_stats:
        if stats["count"] == 0:
            continue
        exe_time = stats["exe_time_ms"]["mean"]
        baseline_time = baseline["exe_time_ms"]["mean"]
        speedup = baseline_time / exe_time if exe_time > 0 else 0
        lines.append(
            f"| {stats['mode']:10} | {exe_time:13.2f} | {speedup:7.2f}x | {stats['iterations']['mean']:10.1f} | {stats['converged_pct']:8.1f}% |"
        )

    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_profile.py <profile_dir>", file=sys.stderr)
        sys.exit(1)

    profile_dir = Path(sys.argv[1])
    if not profile_dir.exists():
        print(f"Error: Directory not found: {profile_dir}", file=sys.stderr)
        sys.exit(1)

    all_stats = []

    # Process each mode directory
    for mode_dir in sorted(profile_dir.iterdir()):
        if not mode_dir.is_dir():
            continue

        mode = mode_dir.name
        debug_file = mode_dir / "debug.jsonl"

        if not debug_file.exists():
            print(f"Warning: No debug.jsonl in {mode_dir}", file=sys.stderr)
            continue

        print(f"Processing {mode}...")

        # Parse based on mode
        if mode == "builtin":
            alignments = parse_autoware_debug(debug_file)
        else:
            alignments = parse_cuda_debug(debug_file)

        # Aggregate statistics
        stats = ModeStats(mode=mode)
        for a in alignments:
            if a.exe_time_ms > 0:  # Filter out invalid entries
                stats.add(a)

        summary = stats.summary()
        all_stats.append(summary)

        # Save per-mode stats
        with open(mode_dir / "stats.json", "w") as f:
            json.dump(summary, f, indent=2)

    # Generate summary report
    report_lines = [
        "NDT Profiling Report",
        "=" * 60,
        f"Profile directory: {profile_dir}",
        "",
    ]

    for stats in all_stats:
        report_lines.append(format_summary(stats))

    report_lines.append(format_comparison(all_stats))

    report = "\n".join(report_lines)

    # Write report
    summary_file = profile_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(report)

    # Also write JSON summary
    json_file = profile_dir / "summary.json"
    with open(json_file, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(report)


if __name__ == "__main__":
    main()
