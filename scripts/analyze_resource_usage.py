#!/usr/bin/env python3
"""
Analyze resource usage from play_launch metrics.csv files.

Compares CPU, memory, and I/O usage between CUDA and Autoware NDT implementations.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Optional


def parse_metrics_csv(filepath: Path) -> list[dict]:
    """Parse a metrics.csv file into a list of records."""
    records = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            record = {}
            for key, value in row.items():
                if value == '':
                    record[key] = None
                elif key in ('timestamp', 'state'):
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


def analyze_metrics(records: list[dict]) -> dict:
    """Compute statistics from metrics records."""
    if not records:
        return {}

    # Filter out initial startup (first few seconds might be unstable)
    # Use all records for now

    cpu_values = [r['cpu_percent'] for r in records if r.get('cpu_percent') is not None]
    rss_values = [r['rss_bytes'] for r in records if r.get('rss_bytes') is not None]
    vms_values = [r['vms_bytes'] for r in records if r.get('vms_bytes') is not None]
    threads_values = [r['num_threads'] for r in records if r.get('num_threads') is not None]

    # I/O rates
    read_rates = [r['total_read_rate_bps'] for r in records if r.get('total_read_rate_bps') is not None]
    write_rates = [r['total_write_rate_bps'] for r in records if r.get('total_write_rate_bps') is not None]

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

    return {
        'cpu_percent': stats(cpu_values),
        'rss_mb': stats([v / (1024 * 1024) for v in rss_values]),
        'vms_mb': stats([v / (1024 * 1024) for v in vms_values]),
        'threads': stats(threads_values),
        'read_rate_mbps': stats([v / (1024 * 1024) for v in read_rates] if read_rates else []),
        'write_rate_mbps': stats([v / (1024 * 1024) for v in write_rates] if write_rates else []),
        'duration_samples': len(records),
    }


def parse_tegrastats_log(filepath: Path) -> list[dict]:
    """Parse tegrastats output log file.

    Format example:
    01-30-2026 18:09:55 RAM 14384/62841MB (lfb 303x4MB) SWAP 0/31420MB (cached 0MB)
    CPU [2%@729,3%@729,...] GR3D_FREQ 0% cpu@48.75C ...
    VDD_GPU_SOC 3192mW/3192mW VDD_CPU_CV 1596mW/1596mW VIN_SYS_5V0 5048mW/5048mW
    """
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = {}

            # Find RAM usage: "RAM 14384/62841MB"
            import re
            ram_match = re.search(r'RAM (\d+)/(\d+)MB', line)
            if ram_match:
                record['ram_used_mb'] = int(ram_match.group(1))
                record['ram_total_mb'] = int(ram_match.group(2))

            # Find GPU utilization: "GR3D_FREQ 0%" or "GR3D_FREQ 50%@1300"
            gpu_match = re.search(r'GR3D_FREQ (\d+)%', line)
            if gpu_match:
                record['gpu_percent'] = int(gpu_match.group(1))

            # Find power values: "VDD_GPU_SOC 3192mW/3192mW"
            gpu_power_match = re.search(r'VDD_GPU_SOC (\d+)mW', line)
            if gpu_power_match:
                record['gpu_power_mw'] = int(gpu_power_match.group(1))

            cpu_power_match = re.search(r'VDD_CPU_CV (\d+)mW', line)
            if cpu_power_match:
                record['cpu_power_mw'] = int(cpu_power_match.group(1))

            total_power_match = re.search(r'VIN_SYS_5V0 (\d+)mW', line)
            if total_power_match:
                record['total_power_mw'] = int(total_power_match.group(1))

            # Find CPU temperatures: "cpu@48.75C"
            temp_match = re.search(r'cpu@([\d.]+)C', line)
            if temp_match:
                record['cpu_temp_c'] = float(temp_match.group(1))

            if record:
                records.append(record)

    return records


def analyze_tegrastats(records: list[dict]) -> dict:
    """Compute GPU statistics from tegrastats records."""
    if not records:
        return {}

    def stats(values: list) -> dict:
        if not values:
            return {'mean': None, 'max': None, 'min': None}
        n = len(values)
        mean = sum(values) / n
        return {
            'mean': mean,
            'max': max(values),
            'min': min(values),
            'count': n
        }

    gpu_values = [r['gpu_percent'] for r in records if r.get('gpu_percent') is not None]
    gpu_power = [r['gpu_power_mw'] for r in records if r.get('gpu_power_mw') is not None]
    cpu_power = [r['cpu_power_mw'] for r in records if r.get('cpu_power_mw') is not None]
    total_power = [r['total_power_mw'] for r in records if r.get('total_power_mw') is not None]
    ram_used = [r['ram_used_mb'] for r in records if r.get('ram_used_mb') is not None]
    cpu_temp = [r['cpu_temp_c'] for r in records if r.get('cpu_temp_c') is not None]

    return {
        'gpu_percent': stats(gpu_values),
        'gpu_power_mw': stats(gpu_power),
        'cpu_power_mw': stats(cpu_power),
        'total_power_mw': stats(total_power),
        'system_ram_mb': stats(ram_used),
        'cpu_temp_c': stats(cpu_temp),
    }


def find_play_log_dir(run_dir: str) -> Optional[Path]:
    """Find the play_log directory for a given run timestamp."""
    play_log_base = Path('/home/jetson/cuda_ndt_matcher/play_log')

    # Direct path
    if Path(run_dir).exists():
        return Path(run_dir)

    # Try as timestamp
    for d in play_log_base.iterdir():
        if d.is_dir() and run_dir in d.name:
            return d

    return None


def identify_run_type(run_dir: Path) -> str:
    """Identify if a run is CUDA or Autoware based on cmdline."""
    cmdline_path = run_dir / 'node' / 'ndt_scan_matcher' / 'cmdline'
    if cmdline_path.exists():
        cmdline = cmdline_path.read_text()
        if 'cuda_ndt_matcher' in cmdline:
            return 'cuda'
        elif 'autoware_ndt_scan_matcher' in cmdline:
            return 'autoware'
    return 'unknown'


def compare_runs(cuda_dir: Path, autoware_dir: Path, verbose: bool = False) -> dict:
    """Compare resource usage between CUDA and Autoware runs."""

    # Verify run types
    cuda_type = identify_run_type(cuda_dir)
    autoware_type = identify_run_type(autoware_dir)

    if cuda_type != 'cuda':
        print(f"Warning: Expected CUDA run but got {cuda_type} for {cuda_dir}", file=sys.stderr)
    if autoware_type != 'autoware':
        print(f"Warning: Expected Autoware run but got {autoware_type} for {autoware_dir}", file=sys.stderr)

    cuda_metrics_path = cuda_dir / 'node' / 'ndt_scan_matcher' / 'metrics.csv'
    autoware_metrics_path = autoware_dir / 'node' / 'ndt_scan_matcher' / 'metrics.csv'

    if not cuda_metrics_path.exists():
        print(f"Error: CUDA metrics not found: {cuda_metrics_path}", file=sys.stderr)
        return {}
    if not autoware_metrics_path.exists():
        print(f"Error: Autoware metrics not found: {autoware_metrics_path}", file=sys.stderr)
        return {}

    cuda_records = parse_metrics_csv(cuda_metrics_path)
    autoware_records = parse_metrics_csv(autoware_metrics_path)

    cuda_stats = analyze_metrics(cuda_records)
    autoware_stats = analyze_metrics(autoware_records)

    # Check for tegrastats logs
    cuda_tegra = cuda_dir / 'tegrastats.log'
    autoware_tegra = autoware_dir / 'tegrastats.log'

    cuda_gpu = analyze_tegrastats(parse_tegrastats_log(cuda_tegra)) if cuda_tegra.exists() else {}
    autoware_gpu = analyze_tegrastats(parse_tegrastats_log(autoware_tegra)) if autoware_tegra.exists() else {}

    return {
        'cuda': {
            'metrics': cuda_stats,
            'gpu': cuda_gpu,
            'dir': str(cuda_dir),
        },
        'autoware': {
            'metrics': autoware_stats,
            'gpu': autoware_gpu,
            'dir': str(autoware_dir),
        }
    }


def print_comparison(comparison: dict, output_format: str = 'table'):
    """Print comparison results."""
    cuda = comparison.get('cuda', {}).get('metrics', {})
    autoware = comparison.get('autoware', {}).get('metrics', {})

    if output_format == 'json':
        print(json.dumps(comparison, indent=2))
        return

    print("=" * 70)
    print(" NDT Resource Usage Comparison")
    print("=" * 70)
    print()
    print(f"CUDA run:     {comparison.get('cuda', {}).get('dir', 'N/A')}")
    print(f"Autoware run: {comparison.get('autoware', {}).get('dir', 'N/A')}")
    print()

    # CPU comparison
    print("-" * 70)
    print(" CPU Usage")
    print("-" * 70)
    cuda_cpu = cuda.get('cpu_percent', {})
    aw_cpu = autoware.get('cpu_percent', {})

    if cuda_cpu.get('mean') is not None and aw_cpu.get('mean') is not None:
        savings = aw_cpu['mean'] - cuda_cpu['mean']
        savings_pct = (savings / aw_cpu['mean'] * 100) if aw_cpu['mean'] > 0 else 0
        print(f"{'Metric':<20} {'CUDA':>12} {'Autoware':>12} {'Savings':>12}")
        print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")
        print(f"{'Mean (%)':<20} {cuda_cpu['mean']:>12.1f} {aw_cpu['mean']:>12.1f} {savings:>+12.1f}")
        print(f"{'Max (%)':<20} {cuda_cpu['max']:>12.1f} {aw_cpu['max']:>12.1f}")
        print(f"{'Samples':<20} {cuda_cpu.get('count', 'N/A'):>12} {aw_cpu.get('count', 'N/A'):>12}")
        print()
        print(f"  CPU savings: {savings:.1f}% ({savings_pct:.1f}% reduction)")
    else:
        print("  CPU data not available")
    print()

    # Memory comparison
    print("-" * 70)
    print(" Memory Usage (RSS)")
    print("-" * 70)
    cuda_mem = cuda.get('rss_mb', {})
    aw_mem = autoware.get('rss_mb', {})

    if cuda_mem.get('mean') is not None and aw_mem.get('mean') is not None:
        savings = aw_mem['mean'] - cuda_mem['mean']
        savings_pct = (savings / aw_mem['mean'] * 100) if aw_mem['mean'] > 0 else 0
        print(f"{'Metric':<20} {'CUDA':>12} {'Autoware':>12} {'Savings':>12}")
        print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")
        print(f"{'Mean (MB)':<20} {cuda_mem['mean']:>12.1f} {aw_mem['mean']:>12.1f} {savings:>+12.1f}")
        print(f"{'Max (MB)':<20} {cuda_mem['max']:>12.1f} {aw_mem['max']:>12.1f}")
        print()
        print(f"  Memory savings: {savings:.1f} MB ({savings_pct:.1f}% reduction)")
    else:
        print("  Memory data not available")
    print()

    # Thread comparison
    print("-" * 70)
    print(" Thread Count")
    print("-" * 70)
    cuda_threads = cuda.get('threads', {})
    aw_threads = autoware.get('threads', {})

    if cuda_threads.get('mean') is not None and aw_threads.get('mean') is not None:
        print(f"{'Metric':<20} {'CUDA':>12} {'Autoware':>12}")
        print(f"{'-'*20} {'-'*12} {'-'*12}")
        print(f"{'Mean':<20} {cuda_threads['mean']:>12.1f} {aw_threads['mean']:>12.1f}")
        print(f"{'Max':<20} {cuda_threads['max']:>12.0f} {aw_threads['max']:>12.0f}")
    else:
        print("  Thread data not available")
    print()

    # GPU comparison (if available)
    cuda_gpu = comparison.get('cuda', {}).get('gpu', {})
    aw_gpu = comparison.get('autoware', {}).get('gpu', {})

    if cuda_gpu or aw_gpu:
        print("-" * 70)
        print(" GPU & Power Usage (from tegrastats)")
        print("-" * 70)

        def fmt(val):
            if val is None:
                return 'N/A'
            return f"{val:.1f}" if isinstance(val, float) else str(val)

        print(f"{'Metric':<25} {'CUDA':>12} {'Autoware':>12} {'Diff':>12}")
        print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12}")

        # GPU utilization
        cuda_gpu_pct = cuda_gpu.get('gpu_percent', {}).get('mean')
        aw_gpu_pct = aw_gpu.get('gpu_percent', {}).get('mean')
        diff = ''
        if cuda_gpu_pct is not None and aw_gpu_pct is not None:
            diff = f"{cuda_gpu_pct - aw_gpu_pct:+.1f}"
        print(f"{'GPU Util Mean (%)':<25} {fmt(cuda_gpu_pct):>12} {fmt(aw_gpu_pct):>12} {diff:>12}")

        cuda_gpu_max = cuda_gpu.get('gpu_percent', {}).get('max')
        aw_gpu_max = aw_gpu.get('gpu_percent', {}).get('max')
        print(f"{'GPU Util Max (%)':<25} {fmt(cuda_gpu_max):>12} {fmt(aw_gpu_max):>12}")

        # GPU power
        cuda_gpu_pw = cuda_gpu.get('gpu_power_mw', {}).get('mean')
        aw_gpu_pw = aw_gpu.get('gpu_power_mw', {}).get('mean')
        diff = ''
        if cuda_gpu_pw is not None and aw_gpu_pw is not None:
            diff = f"{cuda_gpu_pw - aw_gpu_pw:+.0f}"
        print(f"{'GPU Power Mean (mW)':<25} {fmt(cuda_gpu_pw):>12} {fmt(aw_gpu_pw):>12} {diff:>12}")

        # CPU power
        cuda_cpu_pw = cuda_gpu.get('cpu_power_mw', {}).get('mean')
        aw_cpu_pw = aw_gpu.get('cpu_power_mw', {}).get('mean')
        diff = ''
        if cuda_cpu_pw is not None and aw_cpu_pw is not None:
            diff = f"{cuda_cpu_pw - aw_cpu_pw:+.0f}"
        print(f"{'CPU Power Mean (mW)':<25} {fmt(cuda_cpu_pw):>12} {fmt(aw_cpu_pw):>12} {diff:>12}")

        # Total power
        cuda_total_pw = cuda_gpu.get('total_power_mw', {}).get('mean')
        aw_total_pw = aw_gpu.get('total_power_mw', {}).get('mean')
        diff = ''
        if cuda_total_pw is not None and aw_total_pw is not None:
            diff = f"{cuda_total_pw - aw_total_pw:+.0f}"
            savings_pct = (aw_total_pw - cuda_total_pw) / aw_total_pw * 100 if aw_total_pw > 0 else 0
        print(f"{'Total Power Mean (mW)':<25} {fmt(cuda_total_pw):>12} {fmt(aw_total_pw):>12} {diff:>12}")

        # System RAM
        cuda_ram = cuda_gpu.get('system_ram_mb', {}).get('mean')
        aw_ram = aw_gpu.get('system_ram_mb', {}).get('mean')
        diff = ''
        if cuda_ram is not None and aw_ram is not None:
            diff = f"{cuda_ram - aw_ram:+.0f}"
        print(f"{'System RAM Mean (MB)':<25} {fmt(cuda_ram):>12} {fmt(aw_ram):>12} {diff:>12}")

        # Samples
        cuda_samples = cuda_gpu.get('gpu_percent', {}).get('count', 'N/A')
        aw_samples = aw_gpu.get('gpu_percent', {}).get('count', 'N/A')
        print(f"{'Samples':<25} {fmt(cuda_samples):>12} {fmt(aw_samples):>12}")

        if cuda_total_pw is not None and aw_total_pw is not None:
            print()
            power_diff = aw_total_pw - cuda_total_pw
            savings_pct = power_diff / aw_total_pw * 100 if aw_total_pw > 0 else 0
            print(f"  Power savings: {power_diff:.0f} mW ({savings_pct:.1f}% reduction)")
        print()
    else:
        print("-" * 70)
        print(" GPU & Power Usage")
        print("-" * 70)
        print("  No tegrastats data found.")
        print("  Run profiling with: just profile-resource")
        print()

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Analyze NDT resource usage')
    parser.add_argument('--cuda', help='CUDA play_log directory or timestamp')
    parser.add_argument('--autoware', help='Autoware play_log directory or timestamp')
    parser.add_argument('--latest', action='store_true', help='Use latest runs from play_log')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    play_log_base = Path('/home/jetson/cuda_ndt_matcher/play_log')

    if args.latest:
        # Find two most recent runs (should be one CUDA, one Autoware)
        runs = sorted([d for d in play_log_base.iterdir() if d.is_dir() and d.name != 'latest'],
                      key=lambda x: x.name, reverse=True)

        cuda_dir = None
        autoware_dir = None

        for run_dir in runs[:4]:  # Check last 4 runs
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
            print("Error: Could not find both CUDA and Autoware runs in recent play_logs", file=sys.stderr)
            sys.exit(1)
    else:
        if not args.cuda or not args.autoware:
            parser.error("Either --latest or both --cuda and --autoware required")

        cuda_dir = find_play_log_dir(args.cuda)
        autoware_dir = find_play_log_dir(args.autoware)

        if not cuda_dir:
            print(f"Error: CUDA directory not found: {args.cuda}", file=sys.stderr)
            sys.exit(1)
        if not autoware_dir:
            print(f"Error: Autoware directory not found: {args.autoware}", file=sys.stderr)
            sys.exit(1)

    comparison = compare_runs(cuda_dir, autoware_dir, verbose=args.verbose)
    print_comparison(comparison, output_format='json' if args.json else 'table')


if __name__ == '__main__':
    main()
