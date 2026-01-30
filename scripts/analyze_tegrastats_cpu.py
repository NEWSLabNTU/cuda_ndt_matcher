#!/usr/bin/env python3
"""
Analyze total CPU usage from tegrastats logs.

tegrastats provides per-core CPU utilization, which we can sum for total system CPU.
"""

import argparse
import re
import sys
from pathlib import Path


def parse_tegrastats_cpu(filepath: Path) -> list[dict]:
    """Parse tegrastats log for CPU utilization.

    Format: CPU [2%@729,3%@729,0%@729,...] - 12 cores
    """
    records = []
    cpu_pattern = re.compile(r'CPU \[([\d%@,]+)\]')

    with open(filepath) as f:
        for line in f:
            match = cpu_pattern.search(line)
            if match:
                cpu_str = match.group(1)
                # Parse "2%@729,3%@729,..." format
                cores = []
                for core in cpu_str.split(','):
                    pct = core.split('%')[0]
                    try:
                        cores.append(int(pct))
                    except ValueError:
                        pass

                if cores:
                    records.append({
                        'per_core': cores,
                        'total_percent': sum(cores),
                        'core_count': len(cores),
                        'avg_per_core': sum(cores) / len(cores),
                    })

    return records


def analyze_cpu_records(records: list[dict]) -> dict:
    """Compute statistics from CPU records."""
    if not records:
        return {}

    def stats(values: list) -> dict:
        if not values:
            return {'mean': None, 'max': None, 'min': None, 'std': None}
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0
        return {
            'mean': mean,
            'max': max(values),
            'min': min(values),
            'std': variance ** 0.5,
            'count': n
        }

    total_pcts = [r['total_percent'] for r in records]
    avg_per_core = [r['avg_per_core'] for r in records]

    return {
        'total_percent': stats(total_pcts),
        'avg_per_core': stats(avg_per_core),
        'core_count': records[0]['core_count'] if records else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze tegrastats CPU usage')
    parser.add_argument('--cuda', required=True, help='CUDA tegrastats log')
    parser.add_argument('--autoware', required=True, help='Autoware tegrastats log')

    args = parser.parse_args()

    cuda_records = parse_tegrastats_cpu(Path(args.cuda))
    autoware_records = parse_tegrastats_cpu(Path(args.autoware))

    cuda_stats = analyze_cpu_records(cuda_records)
    autoware_stats = analyze_cpu_records(autoware_records)

    print("=" * 70)
    print(" System-Wide CPU Usage (tegrastats, all 12 cores)")
    print("=" * 70)
    print()
    print(f"CUDA samples:     {len(cuda_records)}")
    print(f"Autoware samples: {len(autoware_records)}")
    print()

    cuda_total = cuda_stats.get('total_percent', {})
    aw_total = autoware_stats.get('total_percent', {})
    core_count = cuda_stats.get('core_count', 12)

    if cuda_total.get('mean') is not None and aw_total.get('mean') is not None:
        savings = aw_total['mean'] - cuda_total['mean']
        savings_pct = (savings / aw_total['mean'] * 100) if aw_total['mean'] > 0 else 0

        print(f"{'Metric':<25} {'CUDA':>12} {'Autoware':>12} {'Diff':>12}")
        print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12}")
        print(f"{'Total CPU Mean (%)':<25} {cuda_total['mean']:>12.1f} {aw_total['mean']:>12.1f} {-savings:>+12.1f}")
        print(f"{'Total CPU Max (%)':<25} {cuda_total['max']:>12.1f} {aw_total['max']:>12.1f}")
        print(f"{'Total CPU Min (%)':<25} {cuda_total['min']:>12.1f} {aw_total['min']:>12.1f}")
        print(f"{'Total CPU Std (%)':<25} {cuda_total['std']:>12.1f} {aw_total['std']:>12.1f}")
        print()

        cuda_cores = cuda_total['mean'] / 100
        aw_cores = aw_total['mean'] / 100
        print(f"  Cores ({core_count} available):")
        print(f"    CUDA mean:     {cuda_total['mean']:.1f}% = {cuda_cores:.2f} cores equivalent")
        print(f"    Autoware mean: {aw_total['mean']:.1f}% = {aw_cores:.2f} cores equivalent")
        print(f"    Freed:         {savings:.1f}% = {savings/100:.2f} cores")
        if savings > 0:
            print(f"    Reduction:     {savings_pct:.1f}%")

    print()

    # Per-core average
    cuda_avg = cuda_stats.get('avg_per_core', {})
    aw_avg = autoware_stats.get('avg_per_core', {})

    if cuda_avg.get('mean') is not None and aw_avg.get('mean') is not None:
        print(f"  Per-core average:")
        print(f"    CUDA:     {cuda_avg['mean']:.1f}%")
        print(f"    Autoware: {aw_avg['mean']:.1f}%")

    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
