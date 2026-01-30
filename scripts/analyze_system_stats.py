#!/usr/bin/env python3
"""
Analyze total system CPU and memory usage from play_launch system_stats.csv files.

Compares overall system load between CUDA and Autoware NDT runs.
"""

import argparse
import csv
import sys
from pathlib import Path


def parse_system_stats(filepath: Path) -> list[dict]:
    """Parse system_stats.csv into list of records."""
    records = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = {}
            for key, value in row.items():
                if value == '':
                    record[key] = None
                elif key == 'timestamp':
                    record[key] = value
                else:
                    try:
                        if '.' in str(value):
                            record[key] = float(value)
                        else:
                            record[key] = int(value)
                    except ValueError:
                        record[key] = value
            records.append(record)
    return records


def analyze_system_stats(records: list[dict]) -> dict:
    """Compute statistics from system stats records."""
    if not records:
        return {}

    def stats(values: list) -> dict:
        if not values:
            return {'mean': None, 'max': None, 'min': None, 'std': None}
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0
        std = variance ** 0.5
        return {
            'mean': mean,
            'max': max(values),
            'min': min(values),
            'std': std,
            'count': n
        }

    cpu_values = [r['cpu_percent'] for r in records if r.get('cpu_percent') is not None]
    used_mem = [r['used_memory_bytes'] for r in records if r.get('used_memory_bytes') is not None]
    avail_mem = [r['available_memory_bytes'] for r in records if r.get('available_memory_bytes') is not None]
    total_mem = records[0].get('total_memory_bytes', 0) if records else 0

    return {
        'cpu_percent': stats(cpu_values),
        'used_memory_gb': stats([v / (1024**3) for v in used_mem]),
        'available_memory_gb': stats([v / (1024**3) for v in avail_mem]),
        'total_memory_gb': total_mem / (1024**3) if total_mem else None,
        'cpu_count': records[0].get('cpu_count') if records else None,
    }


def find_play_log_dir(base_path: Path, timestamp_hint: str = None) -> Path:
    """Find play_log directory, optionally matching a timestamp hint."""
    if timestamp_hint:
        # Try to find matching directory
        for d in base_path.iterdir():
            if d.is_dir() and timestamp_hint in d.name:
                return d
    # Return most recent
    dirs = sorted([d for d in base_path.iterdir() if d.is_dir()],
                  key=lambda x: x.name, reverse=True)
    return dirs[0] if dirs else None


def main():
    parser = argparse.ArgumentParser(description='Analyze system CPU/memory usage')
    parser.add_argument('--cuda', help='CUDA play_log directory')
    parser.add_argument('--autoware', help='Autoware play_log directory')
    parser.add_argument('--latest', action='store_true', help='Use two most recent play_log dirs')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    play_log_base = Path('/home/jetson/cuda_ndt_matcher/play_log')

    if args.latest:
        # Find two most recent directories
        dirs = sorted([d for d in play_log_base.iterdir() if d.is_dir() and d.name != 'latest'],
                      key=lambda x: x.name, reverse=True)

        cuda_dir = None
        autoware_dir = None

        for run_dir in dirs[:4]:
            cmdline_path = run_dir / 'node' / 'ndt_scan_matcher' / 'cmdline'
            if cmdline_path.exists():
                cmdline = cmdline_path.read_text()
                if 'cuda_ndt_matcher' in cmdline and cuda_dir is None:
                    cuda_dir = run_dir
                elif 'autoware_ndt_scan_matcher' in cmdline and autoware_dir is None:
                    autoware_dir = run_dir
            if cuda_dir and autoware_dir:
                break

        if not cuda_dir or not autoware_dir:
            print("Error: Could not find both CUDA and Autoware runs", file=sys.stderr)
            sys.exit(1)
    else:
        cuda_dir = Path(args.cuda) if args.cuda else None
        autoware_dir = Path(args.autoware) if args.autoware else None

    # Load and analyze system stats
    cuda_stats_path = cuda_dir / 'system_stats.csv' if cuda_dir else None
    autoware_stats_path = autoware_dir / 'system_stats.csv' if autoware_dir else None

    cuda_stats = {}
    autoware_stats = {}

    if cuda_stats_path and cuda_stats_path.exists():
        cuda_records = parse_system_stats(cuda_stats_path)
        cuda_stats = analyze_system_stats(cuda_records)

    if autoware_stats_path and autoware_stats_path.exists():
        autoware_records = parse_system_stats(autoware_stats_path)
        autoware_stats = analyze_system_stats(autoware_records)

    if args.json:
        import json
        print(json.dumps({
            'cuda': {'dir': str(cuda_dir), 'stats': cuda_stats},
            'autoware': {'dir': str(autoware_dir), 'stats': autoware_stats},
        }, indent=2))
        return

    # Print comparison
    print("=" * 70)
    print(" Total System Resource Usage Comparison")
    print("=" * 70)
    print()
    print(f"CUDA run:     {cuda_dir}")
    print(f"Autoware run: {autoware_dir}")
    print()

    # CPU comparison
    print("-" * 70)
    print(" Total System CPU Usage (all Autoware nodes)")
    print("-" * 70)
    cuda_cpu = cuda_stats.get('cpu_percent', {})
    aw_cpu = autoware_stats.get('cpu_percent', {})
    cpu_count = cuda_stats.get('cpu_count', 12)

    if cuda_cpu.get('mean') is not None and aw_cpu.get('mean') is not None:
        savings = aw_cpu['mean'] - cuda_cpu['mean']
        savings_pct = (savings / aw_cpu['mean'] * 100) if aw_cpu['mean'] > 0 else 0

        print(f"{'Metric':<20} {'CUDA':>12} {'Autoware':>12} {'Diff':>12}")
        print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")
        print(f"{'Mean (%)':<20} {cuda_cpu['mean']:>12.1f} {aw_cpu['mean']:>12.1f} {-savings:>+12.1f}")
        print(f"{'Max (%)':<20} {cuda_cpu['max']:>12.1f} {aw_cpu['max']:>12.1f}")
        print(f"{'Min (%)':<20} {cuda_cpu['min']:>12.1f} {aw_cpu['min']:>12.1f}")
        print(f"{'Std (%)':<20} {cuda_cpu['std']:>12.1f} {aw_cpu['std']:>12.1f}")
        print(f"{'Samples':<20} {cuda_cpu['count']:>12} {aw_cpu['count']:>12}")
        print()
        print(f"  CPU cores: {cpu_count}")
        print(f"  CUDA mean:     {cuda_cpu['mean']:.1f}% = {cuda_cpu['mean']/100:.2f} cores")
        print(f"  Autoware mean: {aw_cpu['mean']:.1f}% = {aw_cpu['mean']/100:.2f} cores")
        print(f"  Difference:    {savings:.1f}% = {savings/100:.2f} cores freed")
        if savings > 0:
            print(f"  Reduction:     {savings_pct:.1f}%")
    else:
        print("  CPU data not available")
    print()

    # Memory comparison
    print("-" * 70)
    print(" Total System Memory Usage")
    print("-" * 70)
    cuda_mem = cuda_stats.get('used_memory_gb', {})
    aw_mem = autoware_stats.get('used_memory_gb', {})
    total_mem = cuda_stats.get('total_memory_gb', 64)

    if cuda_mem.get('mean') is not None and aw_mem.get('mean') is not None:
        savings = aw_mem['mean'] - cuda_mem['mean']

        print(f"{'Metric':<20} {'CUDA':>12} {'Autoware':>12} {'Diff':>12}")
        print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")
        print(f"{'Mean (GB)':<20} {cuda_mem['mean']:>12.2f} {aw_mem['mean']:>12.2f} {-savings:>+12.2f}")
        print(f"{'Max (GB)':<20} {cuda_mem['max']:>12.2f} {aw_mem['max']:>12.2f}")
        print(f"{'Min (GB)':<20} {cuda_mem['min']:>12.2f} {aw_mem['min']:>12.2f}")
        print()
        print(f"  Total system memory: {total_mem:.1f} GB")
        print(f"  CUDA usage:          {cuda_mem['mean']:.2f} GB ({cuda_mem['mean']/total_mem*100:.1f}%)")
        print(f"  Autoware usage:      {aw_mem['mean']:.2f} GB ({aw_mem['mean']/total_mem*100:.1f}%)")
        print(f"  Difference:          {savings:.2f} GB")
    else:
        print("  Memory data not available")
    print()

    print("=" * 70)


if __name__ == '__main__':
    main()
