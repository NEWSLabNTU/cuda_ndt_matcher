#!/usr/bin/env python3
"""
Profile and compare CUDA NDT vs Autoware NDT performance.

Creates dated log directories with a 'latest' symlink.
Compares release vs debug builds to show debug overhead.

Usage:
    python3 scripts/profile_ndt_comparison.py [LOG_DIR]
    python3 scripts/profile_ndt_comparison.py --help
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PerformanceStats:
    """Performance statistics for one implementation."""
    name: str
    build_type: str  # "release" or "debug"
    num_frames: int
    exe_time_mean: float
    exe_time_std: float
    exe_time_min: float
    exe_time_max: float
    exe_time_median: float
    exe_time_p95: float
    exe_time_p99: float
    iterations_mean: float
    iterations_std: float
    iterations_min: int
    iterations_max: int
    throughput_hz: float
    time_per_iteration_mean: float
    time_per_iteration_std: float


@dataclass
class InitPoseStats:
    """Statistics for initial pose estimation."""
    name: str
    num_inits: int
    total_time_mean: float
    total_time_std: float
    total_time_min: float
    total_time_max: float
    startup_time_mean: float
    guided_time_mean: float
    num_particles_mean: float
    num_startup_mean: float
    per_particle_time_mean: float
    final_score_mean: float
    final_iterations_mean: float
    reliable_count: int
    reliable_percent: float


@dataclass
class DebugOverhead:
    """Debug overhead analysis."""
    release_mean: float
    debug_mean: float
    overhead_ms: float
    overhead_percent: float
    speedup: float


def load_jsonl(path: Path) -> List[Dict]:
    """Load a JSON Lines file (alignment entries only)."""
    entries = []
    if not path.exists():
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    if 'exe_time_ms' in entry and 'total_iterations' in entry:
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
    return entries


def load_init_entries(path: Path) -> List[Dict]:
    """Load init pose entries from a JSON Lines file."""
    entries = []
    if not path.exists():
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    if entry.get('type') == 'init':
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
    return entries


def load_init_to_tracking_entries(path: Path) -> List[Dict]:
    """Load init_to_tracking entries from a JSON Lines file."""
    entries = []
    if not path.exists():
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    if entry.get('type') == 'init_to_tracking':
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
    return entries


def compute_init_stats(name: str, entries: List[Dict]) -> Optional[InitPoseStats]:
    """Compute statistics for init pose entries."""
    if not entries:
        return None

    import numpy as np

    total_times = [e['total_time_ms'] for e in entries]
    startup_times = [e['startup_time_ms'] for e in entries]
    guided_times = [e['guided_time_ms'] for e in entries]
    num_particles = [e['num_particles'] for e in entries]
    num_startup = [e.get('num_startup', 0) for e in entries]
    final_scores = [e['final_score'] for e in entries]
    final_iters = [e.get('final_iterations', 0) for e in entries]
    reliable = [e['reliable'] for e in entries]

    # Per-particle time (average across all particles)
    per_particle_times = []
    for e in entries:
        times = e.get('per_particle_time_ms', [])
        if times:
            per_particle_times.extend(times)

    total_times = np.array(total_times)
    reliable_count = sum(reliable)

    return InitPoseStats(
        name=name,
        num_inits=len(entries),
        total_time_mean=float(np.mean(total_times)),
        total_time_std=float(np.std(total_times)),
        total_time_min=float(np.min(total_times)),
        total_time_max=float(np.max(total_times)),
        startup_time_mean=float(np.mean(startup_times)),
        guided_time_mean=float(np.mean(guided_times)),
        num_particles_mean=float(np.mean(num_particles)),
        num_startup_mean=float(np.mean(num_startup)),
        per_particle_time_mean=float(np.mean(per_particle_times)) if per_particle_times else 0.0,
        final_score_mean=float(np.mean(final_scores)),
        final_iterations_mean=float(np.mean(final_iters)),
        reliable_count=reliable_count,
        reliable_percent=(reliable_count / len(entries)) * 100 if entries else 0.0,
    )


def compute_stats(name: str, entries: List[Dict], build_type: str = "release") -> Optional[PerformanceStats]:
    """Compute performance statistics from alignment entries."""
    if not entries:
        return None

    import numpy as np

    exe_times = [e['exe_time_ms'] for e in entries]
    iterations = [e['total_iterations'] for e in entries]

    time_per_iter = []
    for e in entries:
        if e['total_iterations'] > 0:
            time_per_iter.append(e['exe_time_ms'] / e['total_iterations'])

    exe_times = np.array(exe_times)
    iterations = np.array(iterations)
    time_per_iter = np.array(time_per_iter) if time_per_iter else np.array([0.0])

    return PerformanceStats(
        name=name,
        build_type=build_type,
        num_frames=len(entries),
        exe_time_mean=float(np.mean(exe_times)),
        exe_time_std=float(np.std(exe_times)),
        exe_time_min=float(np.min(exe_times)),
        exe_time_max=float(np.max(exe_times)),
        exe_time_median=float(np.median(exe_times)),
        exe_time_p95=float(np.percentile(exe_times, 95)),
        exe_time_p99=float(np.percentile(exe_times, 99)),
        iterations_mean=float(np.mean(iterations)),
        iterations_std=float(np.std(iterations)),
        iterations_min=int(np.min(iterations)),
        iterations_max=int(np.max(iterations)),
        throughput_hz=float(1000.0 / np.mean(exe_times)) if np.mean(exe_times) > 0 else 0.0,
        time_per_iteration_mean=float(np.mean(time_per_iter)),
        time_per_iteration_std=float(np.std(time_per_iter)),
    )


def analyze_by_iteration_count(entries: List[Dict]) -> Dict[int, Dict]:
    """Analyze performance grouped by iteration count."""
    import numpy as np

    by_iters = {}
    for e in entries:
        iters = e['total_iterations']
        if iters not in by_iters:
            by_iters[iters] = []
        by_iters[iters].append(e['exe_time_ms'])

    result = {}
    for iters, times in sorted(by_iters.items()):
        times = np.array(times)
        result[iters] = {
            'count': len(times),
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
        }
    return result


def analyze_warmup(entries: List[Dict], warmup_frames: int = 10) -> Tuple[float, float]:
    """Analyze warmup effect (first N frames vs rest)."""
    import numpy as np

    if len(entries) <= warmup_frames:
        return 0, 0

    warmup_times = [e['exe_time_ms'] for e in entries[:warmup_frames]]
    steady_times = [e['exe_time_ms'] for e in entries[warmup_frames:]]

    return float(np.mean(warmup_times)), float(np.mean(steady_times))


def compute_debug_overhead(release_stats: PerformanceStats, debug_stats: PerformanceStats) -> DebugOverhead:
    """Compute debug overhead from release vs debug comparison."""
    overhead_ms = debug_stats.exe_time_mean - release_stats.exe_time_mean
    overhead_percent = (overhead_ms / release_stats.exe_time_mean) * 100 if release_stats.exe_time_mean > 0 else 0
    speedup = debug_stats.exe_time_mean / release_stats.exe_time_mean if release_stats.exe_time_mean > 0 else 0

    return DebugOverhead(
        release_mean=release_stats.exe_time_mean,
        debug_mean=debug_stats.exe_time_mean,
        overhead_ms=overhead_ms,
        overhead_percent=overhead_percent,
        speedup=speedup,
    )


def print_stats(stats: PerformanceStats):
    """Print performance statistics."""
    print(f"\n{'='*60}")
    print(f" {stats.name} ({stats.build_type} build)")
    print(f"{'='*60}")
    print(f"  Frames analyzed: {stats.num_frames}")
    print(f"\n  Execution Time (ms):")
    print(f"    Mean:   {stats.exe_time_mean:8.2f}")
    print(f"    Std:    {stats.exe_time_std:8.2f}")
    print(f"    Min:    {stats.exe_time_min:8.2f}")
    print(f"    Max:    {stats.exe_time_max:8.2f}")
    print(f"    Median: {stats.exe_time_median:8.2f}")
    print(f"    P95:    {stats.exe_time_p95:8.2f}")
    print(f"    P99:    {stats.exe_time_p99:8.2f}")
    print(f"\n  Iterations:")
    print(f"    Mean: {stats.iterations_mean:.2f}")
    print(f"    Std:  {stats.iterations_std:.2f}")
    print(f"    Min:  {stats.iterations_min}")
    print(f"    Max:  {stats.iterations_max}")
    print(f"\n  Per-Iteration Time (ms):")
    print(f"    Mean: {stats.time_per_iteration_mean:.2f}")
    print(f"    Std:  {stats.time_per_iteration_std:.2f}")
    print(f"\n  Throughput: {stats.throughput_hz:.1f} Hz")


def print_comparison(cuda_stats: PerformanceStats, autoware_stats: PerformanceStats):
    """Print comparison between CUDA and Autoware."""
    print(f"\n{'='*60}")
    print(f" Performance Comparison ({cuda_stats.build_type} build)")
    print(f"{'='*60}")

    speedup = autoware_stats.exe_time_mean / cuda_stats.exe_time_mean if cuda_stats.exe_time_mean > 0 else 0

    print(f"\n  Execution Time Comparison:")
    print(f"    {'Metric':<20} {'CUDA':>12} {'Autoware':>12} {'Ratio':>10}")
    print(f"    {'-'*54}")
    print(f"    {'Mean (ms)':<20} {cuda_stats.exe_time_mean:>12.2f} {autoware_stats.exe_time_mean:>12.2f} {speedup:>9.2f}x")
    print(f"    {'Median (ms)':<20} {cuda_stats.exe_time_median:>12.2f} {autoware_stats.exe_time_median:>12.2f} {autoware_stats.exe_time_median/cuda_stats.exe_time_median:>9.2f}x")
    print(f"    {'P95 (ms)':<20} {cuda_stats.exe_time_p95:>12.2f} {autoware_stats.exe_time_p95:>12.2f} {autoware_stats.exe_time_p95/cuda_stats.exe_time_p95:>9.2f}x")
    print(f"    {'P99 (ms)':<20} {cuda_stats.exe_time_p99:>12.2f} {autoware_stats.exe_time_p99:>12.2f} {autoware_stats.exe_time_p99/cuda_stats.exe_time_p99:>9.2f}x")

    print(f"\n  Throughput Comparison:")
    print(f"    CUDA:     {cuda_stats.throughput_hz:>6.1f} Hz")
    print(f"    Autoware: {autoware_stats.throughput_hz:>6.1f} Hz")
    print(f"    Ratio:    {cuda_stats.throughput_hz/autoware_stats.throughput_hz:>6.2f}x")

    print(f"\n  Per-Iteration Time:")
    print(f"    CUDA:     {cuda_stats.time_per_iteration_mean:.2f} ms/iter")
    print(f"    Autoware: {autoware_stats.time_per_iteration_mean:.2f} ms/iter")
    iter_speedup = autoware_stats.time_per_iteration_mean / cuda_stats.time_per_iteration_mean if cuda_stats.time_per_iteration_mean > 0 else 0
    print(f"    Ratio:    {iter_speedup:.2f}x")


def print_debug_overhead(cuda_overhead: Optional[DebugOverhead], autoware_overhead: Optional[DebugOverhead]):
    """Print debug overhead analysis."""
    print(f"\n{'='*60}")
    print(f" Debug Overhead Analysis")
    print(f"{'='*60}")

    print(f"\n  {'Implementation':<15} {'Release':>12} {'Debug':>12} {'Overhead':>12} {'Slowdown':>10}")
    print(f"  {'-'*61}")

    if cuda_overhead:
        print(f"  {'CUDA':<15} {cuda_overhead.release_mean:>10.2f}ms {cuda_overhead.debug_mean:>10.2f}ms {cuda_overhead.overhead_ms:>+10.2f}ms {cuda_overhead.speedup:>9.2f}x")
    else:
        print(f"  {'CUDA':<15} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>10}")

    if autoware_overhead:
        print(f"  {'Autoware':<15} {autoware_overhead.release_mean:>10.2f}ms {autoware_overhead.debug_mean:>10.2f}ms {autoware_overhead.overhead_ms:>+10.2f}ms {autoware_overhead.speedup:>9.2f}x")
    else:
        print(f"  {'Autoware':<15} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>10}")

    if cuda_overhead:
        print(f"\n  CUDA Debug Overhead Breakdown:")
        print(f"    Per-frame overhead:     {cuda_overhead.overhead_ms:+.2f} ms ({cuda_overhead.overhead_percent:+.1f}%)")
        print(f"    Throughput impact:      {1000/cuda_overhead.debug_mean:.1f} Hz vs {1000/cuda_overhead.release_mean:.1f} Hz")


def print_init_stats(stats: InitPoseStats):
    """Print initial pose estimation statistics."""
    print(f"\n{'='*60}")
    print(f" Initial Pose Estimation: {stats.name}")
    print(f"{'='*60}")
    print(f"  Init operations: {stats.num_inits}")
    print(f"\n  Total Time (ms):")
    print(f"    Mean:   {stats.total_time_mean:8.2f}")
    print(f"    Std:    {stats.total_time_std:8.2f}")
    print(f"    Min:    {stats.total_time_min:8.2f}")
    print(f"    Max:    {stats.total_time_max:8.2f}")
    print(f"\n  Phase Breakdown:")
    print(f"    Startup (random):  {stats.startup_time_mean:.2f} ms")
    print(f"    Guided (TPE):      {stats.guided_time_mean:.2f} ms")
    print(f"\n  Particles:")
    print(f"    Total:             {stats.num_particles_mean:.1f}")
    print(f"    Startup phase:     {stats.num_startup_mean:.1f}")
    print(f"    Per-particle time: {stats.per_particle_time_mean:.2f} ms")
    print(f"\n  Results:")
    print(f"    Mean final score:  {stats.final_score_mean:.4f}")
    print(f"    Mean iterations:   {stats.final_iterations_mean:.1f}")
    print(f"    Reliable:          {stats.reliable_count}/{stats.num_inits} ({stats.reliable_percent:.1f}%)")


def generate_markdown_report(
    log_dir: Path,
    cuda_release: Optional[PerformanceStats],
    cuda_debug: Optional[PerformanceStats],
    autoware_release: Optional[PerformanceStats],
    autoware_debug: Optional[PerformanceStats],
    cuda_by_iters: Dict,
    autoware_by_iters: Dict,
    cuda_warmup: Tuple[float, float],
    autoware_warmup: Tuple[float, float],
    cuda_init: Optional[InitPoseStats] = None,
    autoware_init: Optional[InitPoseStats] = None,
) -> str:
    """Generate markdown report."""
    md = []
    md.append(f"# NDT Profiling Report")
    md.append(f"")
    md.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"**Log Directory**: `{log_dir}`")
    md.append(f"")

    # Performance Summary
    md.append("## Performance Summary (Release Build)")
    md.append("")
    if cuda_release and autoware_release:
        speedup = autoware_release.exe_time_mean / cuda_release.exe_time_mean if cuda_release.exe_time_mean > 0 else 0
        md.append("| Metric | CUDA | Autoware | Ratio |")
        md.append("|--------|------|----------|-------|")
        md.append(f"| Mean exec time (ms) | {cuda_release.exe_time_mean:.2f} | {autoware_release.exe_time_mean:.2f} | {speedup:.2f}x |")
        md.append(f"| Median exec time (ms) | {cuda_release.exe_time_median:.2f} | {autoware_release.exe_time_median:.2f} | {autoware_release.exe_time_median/cuda_release.exe_time_median:.2f}x |")
        md.append(f"| P95 exec time (ms) | {cuda_release.exe_time_p95:.2f} | {autoware_release.exe_time_p95:.2f} | {autoware_release.exe_time_p95/cuda_release.exe_time_p95:.2f}x |")
        md.append(f"| P99 exec time (ms) | {cuda_release.exe_time_p99:.2f} | {autoware_release.exe_time_p99:.2f} | {autoware_release.exe_time_p99/cuda_release.exe_time_p99:.2f}x |")
        md.append(f"| Throughput (Hz) | {cuda_release.throughput_hz:.1f} | {autoware_release.throughput_hz:.1f} | {cuda_release.throughput_hz/autoware_release.throughput_hz:.2f}x |")
        md.append(f"| Mean iterations | {cuda_release.iterations_mean:.2f} | {autoware_release.iterations_mean:.2f} | - |")
        md.append(f"| Time per iteration (ms) | {cuda_release.time_per_iteration_mean:.2f} | {autoware_release.time_per_iteration_mean:.2f} | {autoware_release.time_per_iteration_mean/cuda_release.time_per_iteration_mean:.2f}x |")
        md.append(f"| Frames analyzed | {cuda_release.num_frames} | {autoware_release.num_frames} | - |")
    else:
        md.append("*Release profiling data not available*")
    md.append("")

    # Debug Overhead
    md.append("## Debug Overhead Analysis")
    md.append("")
    md.append("| Implementation | Release (ms) | Debug (ms) | Overhead | Slowdown |")
    md.append("|----------------|--------------|------------|----------|----------|")

    if cuda_release and cuda_debug:
        overhead = compute_debug_overhead(cuda_release, cuda_debug)
        md.append(f"| CUDA | {overhead.release_mean:.2f} | {overhead.debug_mean:.2f} | +{overhead.overhead_ms:.2f}ms ({overhead.overhead_percent:.1f}%) | {overhead.speedup:.2f}x |")
    else:
        md.append("| CUDA | N/A | N/A | N/A | N/A |")

    if autoware_release and autoware_debug:
        overhead = compute_debug_overhead(autoware_release, autoware_debug)
        md.append(f"| Autoware | {overhead.release_mean:.2f} | {overhead.debug_mean:.2f} | +{overhead.overhead_ms:.2f}ms ({overhead.overhead_percent:.1f}%) | {overhead.speedup:.2f}x |")
    else:
        md.append("| Autoware | N/A | N/A | N/A | N/A |")
    md.append("")

    # Warmup Analysis
    md.append("## Warmup Analysis")
    md.append("")
    md.append("| Phase | CUDA (ms) | Autoware (ms) |")
    md.append("|-------|-----------|---------------|")
    md.append(f"| First 10 frames | {cuda_warmup[0]:.2f} | {autoware_warmup[0]:.2f} |")
    md.append(f"| Steady state | {cuda_warmup[1]:.2f} | {autoware_warmup[1]:.2f} |")
    md.append("")

    # Initial Pose Estimation (if available)
    if cuda_init or autoware_init:
        md.append("## Initial Pose Estimation")
        md.append("")
        md.append("| Metric | CUDA | Autoware |")
        md.append("|--------|------|----------|")

        def fmt(val, fmt_str=".2f"):
            return f"{val:{fmt_str}}" if val is not None else "N/A"

        cuda_total = cuda_init.total_time_mean if cuda_init else None
        aw_total = autoware_init.total_time_mean if autoware_init else None
        md.append(f"| Total time (ms) | {fmt(cuda_total)} | {fmt(aw_total)} |")

        cuda_startup = cuda_init.startup_time_mean if cuda_init else None
        aw_startup = autoware_init.startup_time_mean if autoware_init else None
        md.append(f"| Startup phase (ms) | {fmt(cuda_startup)} | {fmt(aw_startup)} |")

        cuda_guided = cuda_init.guided_time_mean if cuda_init else None
        aw_guided = autoware_init.guided_time_mean if autoware_init else None
        md.append(f"| Guided phase (ms) | {fmt(cuda_guided)} | {fmt(aw_guided)} |")

        cuda_particles = cuda_init.num_particles_mean if cuda_init else None
        aw_particles = autoware_init.num_particles_mean if autoware_init else None
        md.append(f"| Particles evaluated | {fmt(cuda_particles, '.0f')} | {fmt(aw_particles, '.0f')} |")

        cuda_ppt = cuda_init.per_particle_time_mean if cuda_init else None
        aw_ppt = autoware_init.per_particle_time_mean if autoware_init else None
        md.append(f"| Per-particle time (ms) | {fmt(cuda_ppt)} | {fmt(aw_ppt)} |")

        cuda_score = cuda_init.final_score_mean if cuda_init else None
        aw_score = autoware_init.final_score_mean if autoware_init else None
        md.append(f"| Final score | {fmt(cuda_score, '.4f')} | {fmt(aw_score, '.4f')} |")

        cuda_rel = f"{cuda_init.reliable_percent:.1f}%" if cuda_init else None
        aw_rel = f"{autoware_init.reliable_percent:.1f}%" if autoware_init else None
        md.append(f"| Reliable | {cuda_rel or 'N/A'} | {aw_rel or 'N/A'} |")

        cuda_n = cuda_init.num_inits if cuda_init else None
        aw_n = autoware_init.num_inits if autoware_init else None
        md.append(f"| Init operations | {cuda_n or 'N/A'} | {aw_n or 'N/A'} |")
        md.append("")

    # Execution Time by Iteration Count
    md.append("## Execution Time by Iteration Count")
    md.append("")
    md.append("### CUDA")
    md.append("")
    md.append("| Iterations | Count | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |")
    md.append("|------------|-------|-----------|----------|----------|----------|")
    for iters, data in sorted(cuda_by_iters.items()):
        md.append(f"| {iters} | {data['count']} | {data['mean_ms']:.2f} | {data['std_ms']:.2f} | {data['min_ms']:.2f} | {data['max_ms']:.2f} |")
    md.append("")

    md.append("### Autoware")
    md.append("")
    md.append("| Iterations | Count | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |")
    md.append("|------------|-------|-----------|----------|----------|----------|")
    for iters, data in sorted(autoware_by_iters.items()):
        md.append(f"| {iters} | {data['count']} | {data['mean_ms']:.2f} | {data['std_ms']:.2f} | {data['min_ms']:.2f} | {data['max_ms']:.2f} |")
    md.append("")

    return "\n".join(md)


def create_dated_log_dir(base_dir: Path) -> Path:
    """Create a dated log directory and update the 'latest' symlink."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_dir = base_dir / "profiling" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    # Update 'latest' symlink
    latest_link = base_dir / "profiling" / "latest"
    if latest_link.is_symlink():
        latest_link.unlink()
    elif latest_link.exists():
        # If it's a regular file/dir, remove it
        import shutil
        if latest_link.is_dir():
            shutil.rmtree(latest_link)
        else:
            latest_link.unlink()

    # Create relative symlink
    latest_link.symlink_to(timestamp)

    return log_dir


def find_log_files(log_dir: Path) -> Dict[str, Path]:
    """Find profiling log files in a directory."""
    files = {}

    # Check for standard file names
    candidates = {
        'cuda_release': ['ndt_cuda_profiling.jsonl', 'ndt_cuda_release.jsonl'],
        'cuda_debug': ['ndt_cuda_debug.jsonl'],
        'autoware_release': ['ndt_autoware_profiling.jsonl', 'ndt_autoware_release.jsonl'],
        'autoware_debug': ['ndt_autoware_debug.jsonl'],
    }

    for key, names in candidates.items():
        for name in names:
            path = log_dir / name
            if path.exists():
                files[key] = path
                break

    return files


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Profile and compare CUDA NDT vs Autoware NDT performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze latest profiling run
    python3 scripts/profile_ndt_comparison.py

    # Analyze specific dated directory
    python3 scripts/profile_ndt_comparison.py logs/profiling/2026-01-27_143000

    # Create new dated directory (for use by justfile)
    python3 scripts/profile_ndt_comparison.py --create-dir
""")
    parser.add_argument("log_dir", nargs="?", type=Path,
                       help="Log directory to analyze (default: logs/profiling/latest)")
    parser.add_argument("--create-dir", action="store_true",
                       help="Create a new dated log directory and print its path")
    parser.add_argument("--output", "-o", type=Path,
                       help="Output markdown report path (default: <log_dir>/report.md)")
    args = parser.parse_args()

    base_dir = Path("logs")

    # Create new dated directory if requested
    if args.create_dir:
        log_dir = create_dated_log_dir(base_dir)
        print(log_dir)
        return

    # Determine log directory to analyze
    if args.log_dir:
        log_dir = args.log_dir
    else:
        # Try latest symlink first
        latest = base_dir / "profiling" / "latest"
        if latest.exists():
            log_dir = latest.resolve()
        else:
            # Fall back to old logs directory
            log_dir = base_dir

    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        sys.exit(1)

    print(f"Analyzing profiling data from: {log_dir}")

    # Find log files
    files = find_log_files(log_dir)

    if not files:
        print(f"No profiling files found in {log_dir}")
        print("Expected files: ndt_cuda_profiling.jsonl, ndt_autoware_profiling.jsonl")
        sys.exit(1)

    print(f"Found files: {', '.join(f.name for f in files.values())}")

    # Load data
    cuda_release_entries = load_jsonl(files['cuda_release']) if 'cuda_release' in files else []
    cuda_debug_entries = load_jsonl(files['cuda_debug']) if 'cuda_debug' in files else []
    autoware_release_entries = load_jsonl(files['autoware_release']) if 'autoware_release' in files else []
    autoware_debug_entries = load_jsonl(files['autoware_debug']) if 'autoware_debug' in files else []

    # Compute statistics
    cuda_release = compute_stats("CUDA NDT", cuda_release_entries, "release") if cuda_release_entries else None
    cuda_debug = compute_stats("CUDA NDT", cuda_debug_entries, "debug") if cuda_debug_entries else None
    autoware_release = compute_stats("Autoware NDT", autoware_release_entries, "release") if autoware_release_entries else None
    autoware_debug = compute_stats("Autoware NDT", autoware_debug_entries, "debug") if autoware_debug_entries else None

    # Print stats
    if cuda_release:
        print_stats(cuda_release)
    if autoware_release:
        print_stats(autoware_release)

    # Print comparison
    if cuda_release and autoware_release:
        print_comparison(cuda_release, autoware_release)

    # Print debug overhead
    cuda_overhead = compute_debug_overhead(cuda_release, cuda_debug) if cuda_release and cuda_debug else None
    autoware_overhead = compute_debug_overhead(autoware_release, autoware_debug) if autoware_release and autoware_debug else None

    if cuda_overhead or autoware_overhead:
        print_debug_overhead(cuda_overhead, autoware_overhead)

    # Analyze by iteration count (use release data)
    cuda_entries = cuda_release_entries or cuda_debug_entries
    autoware_entries = autoware_release_entries or autoware_debug_entries

    cuda_by_iters = analyze_by_iteration_count(cuda_entries) if cuda_entries else {}
    autoware_by_iters = analyze_by_iteration_count(autoware_entries) if autoware_entries else {}

    cuda_warmup = analyze_warmup(cuda_entries) if cuda_entries else (0, 0)
    autoware_warmup = analyze_warmup(autoware_entries) if autoware_entries else (0, 0)

    # Print iteration breakdown
    if cuda_by_iters or autoware_by_iters:
        print(f"\n{'='*60}")
        print(" Execution Time by Iteration Count")
        print(f"{'='*60}")

        if cuda_by_iters:
            print("\n  CUDA:")
            for iters, data in sorted(cuda_by_iters.items()):
                print(f"    {iters} iters: {data['count']:3d} frames, mean={data['mean_ms']:.2f}ms, std={data['std_ms']:.2f}ms")

        if autoware_by_iters:
            print("\n  Autoware:")
            for iters, data in sorted(autoware_by_iters.items()):
                print(f"    {iters} iters: {data['count']:3d} frames, mean={data['mean_ms']:.2f}ms, std={data['std_ms']:.2f}ms")

    # Print warmup
    if cuda_warmup[0] > 0 or autoware_warmup[0] > 0:
        print(f"\n{'='*60}")
        print(" Warmup Analysis (first 10 frames vs steady state)")
        print(f"{'='*60}")
        if cuda_warmup[0] > 0:
            print(f"  CUDA:     warmup={cuda_warmup[0]:.2f}ms, steady={cuda_warmup[1]:.2f}ms")
        if autoware_warmup[0] > 0:
            print(f"  Autoware: warmup={autoware_warmup[0]:.2f}ms, steady={autoware_warmup[1]:.2f}ms")

    # Load and analyze init pose entries
    cuda_init_entries = load_init_entries(files['cuda_release']) if 'cuda_release' in files else []
    if not cuda_init_entries and 'cuda_debug' in files:
        cuda_init_entries = load_init_entries(files['cuda_debug'])

    autoware_init_entries = load_init_entries(files['autoware_release']) if 'autoware_release' in files else []
    if not autoware_init_entries and 'autoware_debug' in files:
        autoware_init_entries = load_init_entries(files['autoware_debug'])

    cuda_init = compute_init_stats("CUDA NDT", cuda_init_entries) if cuda_init_entries else None
    autoware_init = compute_init_stats("Autoware NDT", autoware_init_entries) if autoware_init_entries else None

    # Print init pose stats
    if cuda_init:
        print_init_stats(cuda_init)
    if autoware_init:
        print_init_stats(autoware_init)

    # Load and display init-to-tracking times
    cuda_init_to_tracking = load_init_to_tracking_entries(files['cuda_release']) if 'cuda_release' in files else []
    if not cuda_init_to_tracking and 'cuda_debug' in files:
        cuda_init_to_tracking = load_init_to_tracking_entries(files['cuda_debug'])

    if cuda_init_to_tracking:
        import numpy as np
        times = [e['elapsed_ms'] for e in cuda_init_to_tracking]
        print(f"\n{'='*60}")
        print(f" Init-to-Tracking Time (CUDA)")
        print(f"{'='*60}")
        print(f"  Count:  {len(times)}")
        print(f"  Mean:   {np.mean(times):.2f} ms")
        print(f"  Std:    {np.std(times):.2f} ms")
        print(f"  Min:    {np.min(times):.2f} ms")
        print(f"  Max:    {np.max(times):.2f} ms")

    # Generate and save markdown report
    md_content = generate_markdown_report(
        log_dir,
        cuda_release, cuda_debug,
        autoware_release, autoware_debug,
        cuda_by_iters, autoware_by_iters,
        cuda_warmup, autoware_warmup,
        cuda_init, autoware_init,
    )

    output_path = args.output or log_dir / "report.md"
    with open(output_path, "w") as f:
        f.write(md_content)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
