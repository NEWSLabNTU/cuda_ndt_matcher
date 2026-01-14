# NDT Profiling Results

This document captures profiling results comparing the CUDA NDT implementation against Autoware's builtin NDT scan matcher.

## Test Environment

- **Date**: 2026-01-14
- **Hardware**: NVIDIA GPU (CUDA enabled)
- **Dataset**: Autoware sample rosbag (~23 seconds of driving data)
- **Map**: sample-map-rosbag (point cloud map)
- **Initial Pose**: user_defined_initial_pose enabled for both runs

## Executive Summary

| Metric             | Autoware (OpenMP) | CUDA NDT     | Ratio            |
|--------------------|-------------------|--------------|------------------|
| **Mean exe time**  | **2.31 ms**       | **12.58 ms** | **5.45x slower** |
| Median exe time    | 2.30 ms           | 15.57 ms     | 6.77x slower     |
| Mean iterations    | 2.73              | 15.64        | 5.73x more       |
| Convergence rate   | 100%              | 53.9%        | -                |
| Hit max iterations | 0%                | 46.1%        | -                |

## Execution Time Comparison

| Metric   | Autoware (OpenMP) | CUDA NDT     | Ratio            |
|----------|-------------------|--------------|------------------|
| **Mean** | **2.31 ms**       | **12.58 ms** | **5.45x slower** |
| Median   | 2.30 ms           | 15.57 ms     | 6.77x slower     |
| Stdev    | 0.66 ms           | 8.09 ms      | -                |
| Min      | 1.11 ms           | 2.00 ms      | 1.80x slower     |
| Max      | 3.97 ms           | 26.81 ms     | 6.75x slower     |
| P95      | 3.34 ms           | 24.57 ms     | 7.35x slower     |
| P99      | 3.56 ms           | 25.30 ms     | 7.11x slower     |

**Sample sizes**: Autoware: 215 alignments, CUDA: 225 alignments

### Execution Time Distribution

**Autoware:**
```
 0- 2ms:   72 ( 33.5%) ################
 2- 5ms:  143 ( 66.5%) #################################
 5-10ms:    0 (  0.0%)
10-15ms:    0 (  0.0%)
```

**CUDA:**
```
 0- 2ms:    0 (  0.0%)
 2- 5ms:   69 ( 30.7%) ###############
 5-10ms:   27 ( 12.0%) ######
10-15ms:   11 (  4.9%) ##
15-20ms:   71 ( 31.6%) ###############
20-30ms:   47 ( 20.9%) ##########
```

## Iteration Analysis

| Metric            | Autoware | CUDA  |
|-------------------|----------|-------|
| Mean iterations   | 2.73     | 15.64 |
| Median iterations | 3.0      | 10.0  |
| Stdev             | 1.41     | 13.65 |
| Min               | 1        | 1     |
| Max               | 6        | 30    |

### Iteration Distribution

**Autoware:**
```
 1- 3:  177 ( 67.0%) #################################
 4- 6:   87 ( 33.0%) ################
 7-10:    0 (  0.0%)
```

**CUDA:**
```
 1- 3:   98 ( 36.2%) ##################
 4- 6:   32 ( 11.8%) #####
 7-10:   10 (  3.7%) #
11-15:    2 (  0.7%)
16-20:    0 (  0.0%)
21-25:    3 (  1.1%)
26-30:  126 ( 46.5%) #######################
```

## Convergence Analysis

| Metric        | Autoware       | CUDA            |
|---------------|----------------|-----------------|
| **Converged** | **264 (100%)** | **146 (53.9%)** |
| MaxIterations | 0 (0%)         | 125 (46.1%)     |

## Score Comparison

| Metric                              | Autoware | CUDA    |
|-------------------------------------|----------|---------|
| Mean score                          | 7786.52  | 4620.60 |
| Median score                        | 8016.22  | 6601.21 |
| Stdev                               | 847.39   | 2783.44 |
| Lab 3 - Localization & SLAM.pptxMin | 5519.44  | 560.53  |
| Max                                 | 8783.90  | 7170.84 |

### Score Evolution (first to last iteration)

| Metric        | Autoware | CUDA  |
|---------------|----------|-------|
| Mean change   | +0.3%    | -9.4% |
| Median change | +0.1%    | -3.7% |

**Note**: CUDA scores decrease during optimization (negative change), while Autoware scores are stable. This indicates potential issues with the CUDA optimizer's step direction or size.

## Pose Estimation Quality

| Metric                          | Autoware | CUDA    |
|---------------------------------|----------|---------|
| Mean initial-to-result distance | 0.078 m  | 0.168 m |
| Median distance                 | 0.101 m  | 0.052 m |
| Max distance                    | 0.266 m  | 1.631 m |
| P95 distance                    | 0.176 m  | 0.812 m |

## Oscillation Analysis (CUDA only)

| Metric                    | Value       |
|---------------------------|-------------|
| Mean oscillation count    | 7.54        |
| Max oscillation count     | 28          |
| Entries with oscillations | 162 (59.8%) |

**Note**: Nearly 60% of CUDA alignments experience oscillation (optimizer direction reversals), contributing to slow convergence.

## Root Cause Analysis

### Primary Issue: Low Convergence Rate

CUDA NDT hits max iterations (30) for 46.1% of alignments, compared to 0% for Autoware. This is the primary cause of the 5.45x performance gap.

**Contributing factors:**

1. **Score degradation during optimization** - CUDA scores decrease on average (-9.4%) while Autoware scores remain stable (+0.3%). This suggests step direction or magnitude issues.

2. **High oscillation rate** - 59.8% of CUDA alignments experience direction reversals, indicating the optimizer is overshooting or the gradients are noisy.

3. **Higher iteration variance** - CUDA iteration count has stdev of 13.65 vs 1.41 for Autoware, showing inconsistent convergence behavior.

### When CUDA Works Well

Looking at the 36.2% of alignments that converge in 1-3 iterations:
- These match Autoware's typical behavior
- Execution times in the 2-5ms range are comparable
- The algorithm itself is sound; the issue is consistency

### Potential Causes

1. **Hessian regularization** - May differ between implementations
2. **Line search behavior** - Step size selection may be suboptimal
3. **Gradient computation** - Numerical precision or formulation differences
4. **Voxel search radius** - May affect correspondence quality

## Recommendations

### High Priority

1. **Debug oscillation behavior**
   - Add detailed logging when direction reverses
   - Compare step sizes between CUDA and Autoware
   - Check if Hessian conditioning differs

2. **Investigate score decrease**
   - The score should improve (decrease in NDT terms = better fit)
   - But score *increasing* during optimization suggests step issues
   - Compare per-iteration score trajectories

3. **Analyze failed convergence cases**
   - What do the 46% non-converging cases have in common?
   - Are they concentrated in specific map regions?
   - Do they correlate with initial pose quality?

### Medium Priority

4. **Profile per-phase timing**
   - Enable `profiling` feature to identify bottlenecks
   - Determine if GPU overhead or algorithm iterations dominate

5. **Compare derivative values**
   - Log gradients/Hessians at each iteration
   - Side-by-side comparison with Autoware values

## Data Files

| File | Description |
|------|-------------|
| `rosbag/builtin_20260114_141715/` | Autoware NDT recorded output |
| `rosbag/cuda_20260114_141818/` | CUDA NDT recorded output |
| `/tmp/ndt_autoware_debug.jsonl` | Autoware iteration debug |
| `/tmp/ndt_cuda_debug.jsonl` | CUDA iteration debug |

## Reproducing Results

```bash
# Clear old debug files
rm -f /tmp/ndt_autoware_debug.jsonl /tmp/ndt_cuda_debug.jsonl

# Run Autoware builtin NDT
just run-builtin

# Run CUDA NDT
just run-cuda

# Analyze results
python3 tmp/analyze_profile.py
python3 tmp/extract_exe_times.py
```

## Conclusion

The CUDA NDT implementation is **5.45x slower** than Autoware's OpenMP implementation. The primary cause is:

1. **Low convergence rate** (53.9% vs 100%) - 46% of alignments hit the 30-iteration limit
2. **Score degradation** during optimization - suggests step size/direction issues
3. **High oscillation rate** (59.8%) - optimizer frequently reverses direction

**Key insight**: When CUDA converges quickly (36% of cases in 1-3 iterations), performance is competitive. The optimization algorithm needs tuning to achieve consistent convergence.

**Next steps**: Focus on understanding why the optimizer oscillates and why scores decrease during iteration. This likely points to Hessian conditioning, step size, or line search differences from Autoware.
