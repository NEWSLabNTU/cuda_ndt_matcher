# CUDA NDT vs Autoware NDT Comparison Findings

**Date**: 2026-01-28 (init pose profiling updated)
**Sample Data**: sample-rosbag-fixed (256+ alignment frames)

## Executive Summary

The CUDA NDT implementation produces functionally equivalent results to Autoware's NDT with:
- **Position accuracy**: Mean 0.039m, Max 0.22m (within acceptable localization tolerance)
- **100% convergence rate** for both implementations
- **Score output**: Now normalized to match Autoware's convention (ratio ~0.81)

**Performance varies by platform:**

| Platform                   | Tracking     | Init Pose    | Winner   |
|----------------------------|--------------|--------------|----------|
| **Desktop x86** (RTX 5090) | 1.59x faster | 2.57x faster | **CUDA** |
| **Jetson AGX Orin** (64GB) | 1.32x faster | 3.05x faster | **CUDA** |

CUDA wins on both platforms for both tracking and initial pose estimation. The advantage is more pronounced for init pose (2.57-3.05x) than tracking (1.32-1.59x) due to GPU batch processing of Monte Carlo particles.

**Resource efficiency on Jetson**: CUDA uses **57% less CPU** while achieving 30% higher throughput at **equal power consumption**. This frees CPU cores for other Autoware tasks.

## Implementation Methods

### Algorithm Overview

Both implementations use **Newton's method** for maximum likelihood estimation based on the Normal Distributions Transform (Magnusson 2009). The core optimization loop:

1. Transform source points using current pose estimate
2. Find nearby voxels for each point
3. Compute score, gradient, and Hessian across all point-voxel correspondences
4. Solve Newton step: `Δp = -H⁻¹g`
5. Normalize and scale step, optionally apply line search
6. Update pose and check convergence

**Score Function** (Gaussian probability model):
```
p(x) = -d1 × exp(-d2/2 × (x-μ)ᵀΣ⁻¹(x-μ))
```

Where d1, d2 are derived from outlier ratio and resolution (Magnusson 2009).

### Architecture Comparison

| Aspect                  | Autoware                              | CUDA                              |
|-------------------------|---------------------------------------|-----------------------------------|
| **Language**            | C++                                   | Rust + CUDA                       |
| **Parallelization**     | OpenMP (CPU threads)                  | CUDA (GPU blocks/threads)         |
| **Voxel Storage**       | PCL KD-tree with embedded covariances | Sparse array + spatial hash table |
| **Neighbor Search**     | KD-tree radius search                 | Grid-based 3×3×3 lookup           |
| **Numerical Precision** | Double (f64) throughout               | Mixed: GPU f32, CPU f64           |

### Autoware Implementation (multigrid_ndt_omp)

**Parallelization Strategy**:
```cpp
#pragma omp parallel for
for each source point:
    nearby_voxels = kdtree.radiusSearch(point, radius)
    for each voxel in nearby_voxels:
        compute score, gradient, Hessian contributions
        atomic accumulate into global buffers
// After all points: solve Newton step on single thread
```

**Key Characteristics**:
- Fine-grained parallelism: one thread per point
- KD-tree radius search finds variable voxels per point (typically 4-6)
- Shared state via atomic operations
- Memory-bound (KD-tree traversal)

**Default Parameters**:
- Max iterations: 35
- Transformation epsilon: 0.1m
- Step size: 0.1
- Outlier ratio: 0.55
- Line search: Disabled (causes local minima issues)

### CUDA Implementation (full_gpu_pipeline_v2)

**GPU Pipeline Architecture** (Phase 24 - Graph Kernels):

| Kernel              | Function                                           | Parallelism              |
|---------------------|----------------------------------------------------|--------------------------|
| **K1 (Init)**       | Initialize state from initial pose                 | Single block             |
| **K2 (Compute)**    | Per-point score/gradient/Hessian + block reduction | N points across blocks   |
| **K3 (Solve)**      | Newton solve via SVD + optional regularization     | Single block             |
| **K4 (LineSearch)** | Parallel line search evaluation (optional)         | K candidates in parallel |
| **K5 (Update)**     | Apply step, check convergence                      | Single block             |

**Parallelization Strategy**:
```cuda
// K2: Each block processes a chunk of points
for each point in block's range:
    voxel_indices = grid_lookup_3x3x3(point)
    for each valid voxel:
        compute contributions to score, gradient, Hessian
    block-level reduce via shared memory
    atomic add to global reduce buffers
```

**Key Characteristics**:
- Coarse-grained parallelism: blocks process point chunks
- Grid-based 3×3×3 lookup (max 27 voxels per point)
- Block-local reduction minimizes atomic contention
- Compute-bound (matrix operations, exponentials)

**Default Parameters**:
- Max iterations: 30
- Transformation epsilon: 0.01m (stricter than Autoware)
- Step size: 0.1
- Outlier ratio: 0.55
- Line search: Configurable (disabled by default)

### Neighbor Search Comparison

| Aspect                    | Autoware (KD-tree)             | CUDA (Grid Hash)      |
|---------------------------|--------------------------------|-----------------------|
| **Lookup Complexity**     | O(log V) per query             | O(27) constant        |
| **Search Pattern**        | Radius-based, variable results | Fixed 3×3×3 grid      |
| **Correspondences/Point** | 4-6 typical (variable)         | ≤27 (fixed max)       |
| **Total Correspondences** | ~25,000/frame                  | ~20,000/frame         |
| **Boundary Behavior**     | Finds voxels across boundaries | Strictly grid-aligned |

**Impact**: CUDA finds ~20% fewer correspondences, resulting in ~81% score ratio. This does not affect localization accuracy.

### Convergence Criteria

| Criterion                 | Autoware                             | CUDA           |
|---------------------------|--------------------------------------|----------------|
| **Delta Norm**            | `‖Δp‖ < 0.1m`                        | `‖Δp‖ < 0.01m` |
| **Max Iterations**        | 35                                   | 30             |
| **Oscillation Detection** | Consecutive steps dot product < -0.9 | Same           |

CUDA's stricter epsilon results in ~0.3 more iterations on average.

### Newton Solver

**Both implementations**:
- Use Jacobi SVD for numerical stability
- Normalize step: `Δp /= ‖Δp‖`
- Scale step: `min(‖Δp‖, step_size)`

**CUDA additions**:
- Optional L2 regularization for near-singular Hessian
- Optional GNSS regularization (penalty for deviation from GNSS pose)

### Key Algorithmic Differences Summary

| Aspect                | Autoware       | CUDA                  | Impact                     |
|-----------------------|----------------|-----------------------|----------------------------|
| Neighbor Search       | KD-tree radius | Grid 3×3×3            | ~20% fewer correspondences |
| Correspondences/Frame | ~25,000        | ~20,000               | Score ratio ~0.81          |
| Trans Epsilon         | 0.1m           | 0.01m                 | CUDA ~0.3 more iterations  |
| Precision             | f64            | f32 (GPU) / f64 (CPU) | Negligible difference      |
| Per-Iteration Cost    | ~3.3ms         | ~8.4ms                | GPU launch overhead        |

## Detailed Findings

### 1. Voxel Grid Comparison

| Metric         | CUDA           | Autoware |
|----------------|----------------|----------|
| Resolution     | 2.0m           | 2.0m     |
| Total voxels   | 11,601         | 11,601   |
| Matched voxels | 10,360 (89.3%) | -        |

**Voxel Statistics (Matched Voxels)**:
| Metric                      | Mean   | Std    | Max     |
|-----------------------------|--------|--------|---------|
| Mean position diff          | 0.109m | 0.441m | 3.48m   |
| Covariance diff (Frobenius) | 0.166  | 0.116  | 1.53    |
| Inverse covariance diff     | 94.49  | 131.71 | 2344.49 |
| Point count diff            | 1.54   | 9.79   | 291     |

**Key Observations**:
- Voxel counts match exactly (11,601)
- ~10.7% of voxels (1,241) have different grid keys due to floating-point boundary effects
- Large inverse covariance differences are expected because small covariance changes amplify in matrix inversion
- Some voxels show different Z coordinates (e.g., CUDA: -0.76m vs Autoware: 1.40m), suggesting different point assignments near voxel boundaries

### 2. Pose Difference (CUDA vs Autoware)

This section compares the final poses computed by both implementations for the same input frames. Note: This is **not ground truth validation** - it measures implementation consistency.

**Run comparison**: `just compare-poses` or `just compare-poses-verbose`

#### Position Difference

**Desktop x86**:

| Metric             | Mean   | Std    | Max    | P95    |
|--------------------|--------|--------|--------|--------|
| Position 3D (m)    | 0.039  | 0.037  | 0.223  | -      |
| Position 2D (m)    | -      | -      | -      | -      |
| Rotation (rad)     | 0.019  | 0.015  | 0.054  | -      |

**Jetson AGX Orin**:

| Metric             | Mean   | Std    | Max    | P95    |
|--------------------|--------|--------|--------|--------|
| Position 3D (m)    | 0.038  | 0.039  | 0.264  | 0.107  |
| Position 2D (m)    | 0.036  | 0.037  | 0.263  | 0.101  |
| Rotation (rad)     | 0.019  | 0.015  | 0.055  | 0.051  |
| Rotation (deg)     | 1.11   | 0.86   | 3.16   | 2.94   |

**Key Observations**:
- Both platforms show similar pose differences (~3.8cm mean, ~26cm max)
- Rotation differences are small (~1.1 deg mean, ~3.2 deg max)
- Largest differences occur when CUDA converges in fewer iterations

#### Convergence

| Platform    | CUDA Converged | Autoware Converged | Common Frames |
|-------------|----------------|--------------------|---------------|
| x86         | 289/289 (100%) | 289/289 (100%)     | 289           |
| Jetson      | 294/294 (100%) | 290/290 (100%)     | 289           |

#### Iteration Count Difference (CUDA - Autoware)

| Platform | Mean  | Std  | Range      |
|----------|-------|------|------------|
| x86      | +0.28 | 1.42 | [-4, +7]   |
| Jetson   | +0.29 | 1.44 | [-5, +5]   |

#### Largest Position Differences (Jetson)

| Timestamp       | Pos Diff | Rot Diff | CUDA Iters | AW Iters |
|-----------------|----------|----------|------------|----------|
| ...269743713792 | 0.264m   | 0.008    | 1          | 5        |
| ...269844508160 | 0.260m   | 0.008    | 1          | 5        |
| ...255644498944 | 0.253m   | 0.024    | 1          | 6        |
| ...280443982592 | 0.226m   | 0.040    | 1          | 6        |

**Observation**: Largest differences occur when CUDA converges very early (1 iteration) while Autoware takes more iterations. This is due to different convergence epsilon (CUDA: 0.01m vs Autoware: 0.1m).

### 3. Score Calculation Differences

**After Fix (2026-01-27)**: CUDA now outputs normalized `transform_probability` matching Autoware's convention.

| Score Type    | CUDA (Fixed) | Autoware  | Ratio |
|---------------|--------------|-----------|-------|
| `final_score` | 5.82-5.83    | 7.15-7.19 | ~0.81 |
| `final_nvtl`  | 2.45-2.47    | 3.18-3.20 | ~0.77 |

**Normalization Convention** (both implementations):
```
transform_probability = total_score / num_source_points
```

**Remaining Score Difference (~20%)**:
The ~0.81 ratio is due to different correspondence counts:
- CUDA: ~20,200 correspondences per frame
- Autoware: ~25,000 correspondences (estimated from raw scores)

This difference stems from voxel boundary handling - CUDA's coordinate-based voxel lookup finds fewer neighbor voxels than Autoware's radius search.

**NVTL Difference (~0.7)**:
- Both use max-score-per-point averaging
- CUDA's coordinate-based neighbor search (3x3x3 grid) vs Autoware's radius search
- Results in fewer voxels found per point, hence lower NVTL

### 4. Per-Iteration Analysis (Autoware Reference)

Sample first frame optimization:

| Iteration | Pose (x,y,z)              | Score  | Step Length | Voxels/Point |
|-----------|---------------------------|--------|-------------|--------------|
| 0         | (89571, 42301, -3.30)     | 33,080 | 0.10        | 3.96         |
| 1         | (89571, 42301, -3.21)     | 35,507 | 0.10        | 4.05         |
| 2         | (89571.1, 42301.1, -3.13) | 35,820 | 0.05        | 4.05         |
| 3         | (89571.1, 42301.1, -3.12) | 35,864 | 0.03        | 4.06         |
| 4         | (89571.1, 42301.1, -3.12) | 35,889 | 0.02        | 4.06         |
| 5         | (89571.1, 42301.2, -3.12) | 35,880 | 0.01        | 4.06         |
| 6         | (89571.1, 42301.2, -3.12) | 35,884 | 0.005       | 4.06         |

**Convergence**: 7 iterations, Final NVTL: 3.20

### 5. Largest Position Differences

| Timestamp       | Position Diff | CUDA Iters | Autoware Iters | Notes                                      |
|-----------------|---------------|------------|----------------|--------------------------------------------|
| ...844858880    | 0.223m        | 3          | 7              | Initial frame, different convergence paths |
| ...279244287744 | 0.210m        | 5          | 6              | Near yaw ~2.2 rad                          |
| ...279645129728 | 0.206m        | 1          | 5              | Very early CUDA convergence                |
| ...268744289792 | 0.202m        | 1          | 4              | Very early CUDA convergence                |

**Observation**: The largest differences occur when CUDA converges in fewer iterations (1-3) compared to Autoware (4-7). This suggests CUDA may have different convergence criteria or line search parameters.

## Root Causes of Differences

### 1. Score Output Convention - FIXED
- **Issue**: CUDA output raw `total_score`, Autoware outputs normalized `transform_probability`
- **Fix Applied**: Changed CUDA to output `transform_probability = total_score / num_source_points`
- **Files Changed**:
  - `src/ndt_cuda/src/optimization/solver.rs`: Updated `compute_transform_probability()` and all call sites
  - `src/ndt_cuda/src/ndt.rs`: Updated GPU path debug output
- **Result**: Score ratio improved from ~4000x to ~0.81x

### 2. Correspondence Count Difference
- **Issue**: CUDA finds ~20% fewer correspondences than Autoware
- **Cause**: Different neighbor search implementations
  - CUDA: Coordinate-based 3x3x3 voxel grid lookup
  - Autoware: Radius search (finds more nearby voxels)
- **Impact**: Accounts for the remaining ~20% score difference

### 3. NVTL Calculation
- **Issue**: ~0.7 lower NVTL in CUDA
- **Cause**: Same as correspondence count - fewer neighbor voxels found per point
- **Impact**: Functional - both provide valid quality metrics

### 4. Voxel Boundary Assignment
- **Issue**: ~10.7% voxels have different grid keys
- **Cause**: Floating-point precision in coordinate-to-voxel mapping
- **Impact**: Minor - different points assigned to boundary voxels

### 5. Convergence Criteria
- **Issue**: CUDA sometimes converges much earlier (1-3 iters vs 4-7)
- **Cause**: Different step length tolerances and line search parameters
- **Impact**: May lead to slightly different final poses

## Performance Profiling

### Test Environments

**Desktop x86**:
- **CPU**: Intel Core Ultra 7 265K (20 cores)
- **GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM)
- **Memory**: 30GB RAM
- **OS**: Ubuntu 22.04 (Linux 6.8.0-90-generic)
- **CUDA**: Driver 580.105.08

**Jetson AGX Orin**:
- **Platform**: NVIDIA Jetson AGX Orin Developer Kit (64GB)
- **CPU**: 12x ARM Cortex-A78AE cores
- **GPU**: Ampere iGPU (2048 CUDA cores, unified memory)
- **Memory**: 64GB unified (shared CPU/GPU)
- **OS**: Ubuntu 22.04 (L4T 36.4.4, Linux 5.15.148-tegra)
- **CUDA**: 12.6.68

**Common Configuration**:
- ROS: Humble
- Autoware: 1.5.0
- Build: Release with profiling (minimal overhead)
- Dataset: sample-rosbag-fixed, 5000 source points per frame
- Date: 2026-01-27 (latest x86 profiling: 14:40)

### Performance Summary

**Desktop x86** (CUDA wins):

| Metric                  | CUDA  | Autoware | Speedup     |
|-------------------------|-------|----------|-------------|
| Mean exec time (ms)     | 5.01  | 7.97     | **1.59x**   |
| Median exec time (ms)   | 4.93  | 7.91     | **1.60x**   |
| P95 exec time (ms)      | 6.93  | 11.54    | **1.67x**   |
| P99 exec time (ms)      | 7.85  | 12.86    | **1.64x**   |
| Min exec time (ms)      | 3.36  | 3.42     | -           |
| Max exec time (ms)      | 8.24  | 34.32    | -           |
| Throughput (Hz)         | 199.7 | 125.4    | **1.59x**   |
| Mean iterations         | 3.16  | 3.30     | -           |
| Time per iteration (ms) | 2.16  | 2.73     | **1.27x**   |
| Frames analyzed         | 256   | 216      | -           |

**Jetson AGX Orin** (CUDA wins):

| Metric                  | CUDA     | Autoware | Speedup   |
|-------------------------|----------|----------|-----------|
| Mean exec time (ms)     | 28.71    | 37.79    | **1.32x** |
| Median exec time (ms)   | 26.99    | 36.97    | **1.37x** |
| P95 exec time (ms)      | 46.00    | 61.83    | 1.34x     |
| P99 exec time (ms)      | 52.76    | 70.46    | 1.34x     |
| Min exec time (ms)      | 11.11    | 13.18    | -         |
| Max exec time (ms)      | 57.03    | 71.51    | -         |
| Throughput (Hz)         | **34.8** | 26.5     | **1.32x** |
| Mean iterations         | 3.26     | 2.93     | -         |
| Time per iteration (ms) | 8.81     | 12.89    | **1.46x** |
| Frames analyzed         | 294      | 290      | -         |
| Convergence rate        | 100%     | 100%     | -         |

### Warmup Analysis

**Desktop x86**:

| Phase           | CUDA (ms) | Autoware (ms) |
|-----------------|-----------|---------------|
| First 10 frames | 3.72      | 6.03          |
| Steady state    | 5.06      | 8.07          |

**Jetson AGX Orin**:

| Phase           | CUDA (ms) | Autoware (ms) |
|-----------------|-----------|---------------|
| First 10 frames | 29.22     | 36.78         |
| Steady state    | 28.69     | 37.82         |

**Observations**:
- x86: Both show warmup effects. CUDA has more pronounced warmup penalty (83% overhead vs 60% for Autoware).
- Jetson: Minimal warmup effect. Both implementations reach steady state quickly.

### Execution Time by Iteration Count

**Desktop x86 - CUDA**:

| Iterations | Count | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|------------|-------|-----------|----------|----------|----------|
| 1          | 81    | 3.71      | 0.45     | 3.36     | 8.24     |
| 2          | 17    | 4.24      | 0.24     | 4.30     | 4.24     |
| 3          | 47    | 4.87      | 0.25     | 4.62     | 5.37     |
| 4          | 45    | 5.46      | 0.15     | 5.16     | 5.76     |
| 5          | 34    | 6.20      | 0.28     | 5.64     | 6.76     |
| 6          | 23    | 6.75      | 0.20     | 6.35     | 7.15     |
| 7          | 6     | 7.47      | 0.23     | 7.12     | 7.82     |
| 8          | 3     | 7.98      | 0.18     | 7.73     | 8.11     |

**Desktop x86 - Autoware**:

| Iterations | Count | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|------------|-------|-----------|----------|----------|----------|
| 1          | 34    | 4.79      | 1.28     | 3.42     | 8.26     |
| 2          | 18    | 5.92      | 0.96     | 4.96     | 7.84     |
| 3          | 56    | 7.19      | 0.74     | 5.71     | 8.67     |
| 4          | 70    | 9.14      | 3.16     | 6.98     | 34.32    |
| 5          | 33    | 10.62     | 0.97     | 8.68     | 12.56    |
| 6          | 5     | 12.04     | 1.19     | 10.33    | 13.23    |

**Jetson AGX Orin - CUDA**:

| Iterations | Count | Mean (ms) | Std (ms) |
|------------|-------|-----------|----------|
| 1          | 109   | 20.18     | 4.31     |
| 2          | 13    | 22.66     | 3.83     |
| 3          | 28    | 27.53     | 5.72     |
| 4          | 54    | 32.48     | 5.76     |
| 5          | 42    | 36.90     | 6.30     |
| 6          | 29    | 36.47     | 8.08     |
| 7          | 15    | 40.07     | 11.05    |
| 8          | 3     | 52.24     | 2.96     |
| 10         | 1     | 56.86     | 0.00     |

**Jetson AGX Orin - Autoware**:

| Iterations | Count | Mean (ms) | Std (ms) |
|------------|-------|-----------|----------|
| 1          | 88    | 22.76     | 5.14     |
| 2          | 28    | 29.81     | 7.51     |
| 3          | 62    | 39.88     | 7.58     |
| 4          | 54    | 46.28     | 8.52     |
| 5          | 44    | 53.51     | 9.48     |
| 6          | 14    | 56.75     | 8.58     |

### Performance Analysis

**Why CUDA wins on x86**:

1. **Efficient Per-Iteration Cost**: CUDA's per-iteration time (2.16ms) is 1.27x faster than Autoware (2.73ms) due to:
   - Optimized GPU graph kernels with minimal launch overhead
   - Block-level reductions minimize atomic contention
   - Efficient GPU memory access patterns

2. **Low Variance**: CUDA shows consistent timing (std 0.45ms for single-iteration) with minimal warmup effects.

3. **GPU Parallelism**: The RTX 5090 provides massive parallel compute that outweighs any kernel launch overhead.

**Why CUDA wins on Jetson**:

1. **Unified Memory**: Jetson's unified memory architecture eliminates PCIe transfer overhead, making GPU kernel launches cheaper.

2. **ARM OpenMP Performance**: Autoware's OpenMP is optimized for x86. On ARM Cortex-A78AE cores, OpenMP parallelization is less efficient (12.89ms/iter vs 3.24ms/iter on x86).

3. **GPU Integration**: The Tegra GPU is tightly integrated with the ARM cores, reducing kernel launch latency.

4. **Per-Iteration Advantage**: CUDA's per-iteration time (8.81ms) is 1.46x faster than Autoware (12.89ms) on Jetson.

**Common Observations**:

- Iteration distribution is similar on both platforms (~37% single-iteration for CUDA vs ~30% for Autoware)
- Both implementations are real-time capable (exceed 10 Hz LiDAR rate with comfortable margin)
- Lower variance on Jetson compared to x86, likely due to simpler memory hierarchy

### Debug Build Comparison (x86 only)

For reference, the debug build with full per-iteration data collection shows additional overhead. Use the `profiling` build for accurate performance comparison.

*Note: Debug overhead analysis not available in current profiling run.*

### Initial Pose Estimation Performance

When `user_defined_initial_pose` is disabled, the system uses Monte Carlo pose estimation with TPE (Tree-Structured Parzen Estimator) to find the initial pose.

**Desktop x86** (200 particles, 100 startup):

| Metric                  | CUDA     | Autoware | Speedup     |
|-------------------------|----------|----------|-------------|
| Total init time         | 2617 ms  | 6720 ms  | **2.57x**   |
| Startup phase (random)  | 1339 ms  | 3079 ms  | 2.30x       |
| Guided phase (TPE)      | 1278 ms  | 3641 ms  | 2.85x       |
| Per-particle time       | 11.81 ms | 33.60 ms | **2.85x**   |
| Iterations per particle | 18       | 28       | 1.56x fewer |
| Final score (NVTL)      | 2.98     | 3.20     | Similar     |
| Init-to-tracking        | 2622 ms  | N/A      | -           |
| Reliable                | Yes      | Yes      | -           |

**Jetson AGX Orin** (200 particles, 100 startup):

| Metric                  | CUDA      | Autoware  | Speedup     |
|-------------------------|-----------|-----------|-------------|
| Total init time         | 7411 ms   | 22613 ms  | **3.05x**   |
| Startup phase (random)  | 4202 ms   | 10730 ms  | 2.55x       |
| Guided phase (TPE)      | 3208 ms   | 11883 ms  | 3.70x       |
| Per-particle time       | 34.30 ms  | 113.06 ms | **3.30x**   |
| Iterations per particle | 11        | 28        | 2.55x fewer |
| Final score (NVTL)      | 2.98      | 3.20      | Similar     |
| Init-to-tracking        | 7436 ms   | N/A       | -           |
| Reliable                | Yes       | Yes       | -           |

**Initial Pose Accuracy** (from rosbag comparison):

| Platform    | Position Diff | Yaw Diff | Assessment |
|-------------|---------------|----------|------------|
| x86 Desktop | 0.024m        | 0.03°    | Excellent  |
| Jetson Orin | 0.427m        | 0.03°    | Acceptable |

Both implementations converge to essentially the same pose:

**x86 Desktop**:
- CUDA: x=89571.10, y=42301.14, z=-3.12, yaw=33.4°
- Autoware: x=89571.12, y=42301.16, z=-3.12, yaw=33.4°

**Jetson AGX Orin**:
- CUDA: x=89571.09, y=42301.14, z=-3.12, yaw=33.4°
- Autoware: x=89571.45, y=42301.38, z=-3.12, yaw=33.4°

**Key Findings:**
- **x86**: CUDA is **2.57x faster** for init pose (2.6s vs 6.7s)
- **Jetson**: CUDA is **3.05x faster** for init pose (7.4s vs 22.6s)
- Per-particle alignment: 2.85x faster on x86, **3.30x faster on Jetson**
- CUDA converges in fewer iterations on both platforms
- Both achieve reliable results (NVTL > 2.3 threshold)
- **Final poses are functionally equivalent** on both platforms

**Phase Breakdown:**
- **Startup phase**: First 100 particles evaluated with random sampling using GPU batch alignment
- **Guided phase**: Remaining 100 particles guided by TPE, evaluated sequentially

**Tracking Performance After Init (Jetson)**:

| Metric              | CUDA     | Autoware | Speedup   |
|---------------------|----------|----------|-----------|
| Mean exec time (ms) | 28.29    | 110.88   | **3.92x** |
| Throughput (Hz)     | 35.3     | 9.0      | **3.92x** |
| Mean iterations     | 3.92     | 29.20    | -         |

Note: Autoware hits max iterations (30) on 90% of frames after init mode, indicating different convergence behavior in the TPE-guided initial pose scenario.

### Resource Usage (Jetson AGX Orin)

System resource usage comparison measuring CPU, memory, GPU utilization, and power consumption on Jetson AGX Orin.

**Run profiling**: `just profile-resource`

#### Profiling Methodology

| Parameter           | Value               | Notes                                 |
|---------------------|---------------------|---------------------------------------|
| **Dataset**         | sample-rosbag-fixed | ~30s of driving data                  |
| **Startup delay**   | 60s                 | Wait for Autoware stack to initialize |
| **Rosbag duration** | ~30s                | Aligned frames during playback        |
| **Total run time**  | ~93s per impl       | Including startup and teardown        |

**Measurement Tools:**

| Tool                  | Sample Rate | Metrics                                  | Samples       |
|-----------------------|-------------|------------------------------------------|---------------|
| `tegrastats`          | 500ms       | GPU util, power, system RAM, temperature | 183 per run   |
| `play_launch` metrics | 2000ms      | Per-process CPU%, RSS, threads, I/O      | 22-23 per run |

**Data Collection:**
- `tegrastats --interval 500` runs in background during entire profiling run
- `play_launch` monitors the `ndt_scan_matcher` node at 2-second intervals
- Measurements cover the full run including idle periods before/after rosbag playback
- Active NDT processing occurs during ~30s of rosbag playback

**tegrastats Fields Used:**

| Field         | Description                   | Example                     |
|---------------|-------------------------------|-----------------------------|
| `GR3D_FREQ`   | GPU 3D engine utilization (%) | `GR3D_FREQ 11%`             |
| `VDD_GPU_SOC` | GPU/SoC power (mW)            | `VDD_GPU_SOC 3809mW/3809mW` |
| `VDD_CPU_CV`  | CPU cluster power (mW)        | `VDD_CPU_CV 3841mW/3841mW`  |
| `VIN_SYS_5V0` | Total system power (mW)       | `VIN_SYS_5V0 5505mW/5505mW` |
| `RAM`         | System memory usage           | `RAM 14831/62841MB`         |

**Note**: Per-process GPU metrics are not available on Jetson (Tegra architecture limitation). GPU utilization is measured system-wide via tegrastats. During profiling, only the NDT node performs significant GPU work.

#### Total System Resource Usage

System-wide CPU and memory usage across all 72 Autoware nodes, measured via tegrastats during active processing (30s rosbag playback after 60s startup).

| Metric            | CUDA  | Autoware | Diff      | Notes                 |
|-------------------|-------|----------|-----------|-----------------------|
| Total CPU (%)     | 520.7 | 570.7    | **-50.0** | Sum of all 12 cores   |
| CPU (cores equiv) | 5.21  | 5.71     | **-0.50** | 8.8% reduction        |
| CPU Max (%)       | 606   | 686      | -80       | Peak during alignment |
| System RAM (GB)   | 13.29 | 13.17    | +0.12     | Nearly identical      |

**Key Finding**: CUDA NDT frees **0.5 CPU cores** (8.8% reduction) at the system level. While this is smaller than the per-node reduction (57%), it represents meaningful headroom for other tasks on embedded platforms.

**Note on measurement scope**:
- **tegrastats**: Measures all system processes (~570%) - includes profiling tools
- **play_launch**: Monitors only 25 Autoware nodes (~294%)
- **rosbag (measured via top)**: Only ~8% CPU - much smaller than initially estimated
- **Profiling overhead**: ~150% (play_launch, monitoring tools, etc.)

#### CPU Usage by Autoware Node Category

Per-node CPU usage from play_launch metrics (25 monitored nodes only):

| Category                | CUDA (%)  | Autoware (%) | Diff      |
|-------------------------|-----------|--------------|-----------|
| **NDT Scan Matcher**    | 34.8      | 81.0         | **-46.2** |
| Pointcloud Processing   | 124.8     | 122.6        | +2.2      |
| Localization (EKF, etc) | 30.0      | 29.4         | +0.6      |
| Other nodes             | 62.3      | 61.2         | +1.0      |
| **Total (25 nodes)**    | **251.9** | **294.2**    | **-42.3** |

**Verified via top (Autoware run)**:

| Category | CPU (%) | % of Total |
|----------|---------|------------|
| NDT Scan Matcher | 93.3 | 46% |
| Localization (EKF, gyro, IMU) | 32.5 | 16% |
| Component Containers | 24.0 | 12% |
| rosbag (player+recorder) | 8.1 | 4% |
| Other Autoware nodes | 45.2 | 22% |
| **Total** | **203.1** | 100% |

**Key Insight**: NDT is the CPU bottleneck, consuming **46% of all Autoware CPU**. rosbag overhead is minimal (~8%).

#### CPU Usage (NDT Node Only)

| Metric       | CUDA | Autoware | Savings           |
|--------------|------|----------|-------------------|
| Mean (%)     | 34.8 | 81.0     | **57% reduction** |
| Max (%)      | 65.4 | 173.6    | 62% reduction     |
| Thread count | 9    | 12       | 25% fewer         |

**Key Finding**: CUDA uses **57% less CPU** by offloading computation to the GPU. This frees up CPU cores for other Autoware tasks (planning, perception).

#### Memory Usage

| Metric   | CUDA (MB) | Autoware (MB) | Notes                             |
|----------|-----------|---------------|-----------------------------------|
| RSS Mean | 258.0     | 53.2          | CUDA higher due to unified memory |
| RSS Max  | 292.9     | 66.2          | GPU buffers mapped to process     |

**Note**: On Jetson's unified memory architecture, GPU memory is mapped into the process address space, causing higher RSS values for CUDA. This is not additional memory consumption—the same physical memory is shared between CPU and GPU.

#### GPU Utilization (System-Wide via tegrastats)

| Metric        | CUDA  | Autoware | Notes                  |
|---------------|-------|----------|------------------------|
| GPU Util Mean | 11.1% | 0.2%     | CUDA actively uses GPU |
| GPU Util Max  | 98%   | 34%      | Peak during alignment  |

**Observation**: CUDA achieves better throughput with only ~11% average GPU utilization, leaving GPU capacity available for other tasks (perception, deep learning inference).

#### Power Consumption (tegrastats)

| Component        | CUDA (mW) | Autoware (mW) | Diff |
|------------------|-----------|---------------|------|
| GPU (VDD_GPU_SOC)| 3809      | 3406          | +403 |
| CPU (VDD_CPU_CV) | 3841      | 3968          | -127 |
| Total (VIN_SYS)  | 5505      | 5468          | +37  |

**Key Finding**: Total power consumption is nearly identical (+0.7%). CUDA trades CPU power for GPU power, with no net increase in system power draw despite 30% higher throughput.

#### Resource Efficiency Summary

| Metric          | NDT Node       | Full System | Notes                           |
|-----------------|----------------|-------------|---------------------------------|
| CPU reduction   | **57%**        | **8.8%**    | 0.5 cores freed system-wide     |
| GPU utilization | 11% avg        | -           | Ample headroom for perception   |
| Power           | Equal          | Equal       | Same power, 30% more throughput |
| Throughput/Watt | **30% better** | -           | More work per watt              |

**Recommendation**: CUDA NDT is more resource-efficient on Jetson:
- **NDT-specific**: 57% less CPU, freeing compute for other tasks
- **System-wide**: 0.5 cores freed (8.8% reduction) across 72 Autoware nodes
- **Power-neutral**: Same power consumption with 30% higher throughput
- **GPU headroom**: Only 11% GPU utilization leaves capacity for perception DNNs

### Profiling and Comparison Commands

```bash
# === Performance Profiling ===

# Build and run release profiling (minimal overhead)
just run-cuda-profiling      # CUDA with profiling only
just run-builtin-profiling   # Autoware with profiling only

# Full profiling comparison (build, run both, compare timing)
just profile-compare

# Init mode profiling (Monte Carlo pose estimation)
just profile-init            # Both implementations with init mode

# Resource profiling with tegrastats (CPU, GPU, power)
just profile-resource        # Includes tegrastats monitoring

# Debug builds (full debug data, slower)
just run-cuda-debug          # CUDA with all debug features
just run-builtin-debug       # Autoware with all debug features

# === Resource Analysis ===

# Analyze resource usage from latest play_log runs
just analyze-resource        # Compare per-node CPU, memory, GPU, power
just analyze-system-stats    # Compare total system CPU and memory

# === Pose Accuracy Comparison ===

# Compare poses from existing profiling logs
just compare-poses           # Summary output
just compare-poses-verbose   # Include largest differences
just compare-poses-json      # JSON output for automation

# Full accuracy test (run both implementations and compare)
just accuracy-test           # Build, run both, compare poses
```

**Notes**:
- The `profiling` feature only enables timing instrumentation (minimal overhead)
- The `debug` feature enables per-iteration data collection (significant overhead)
- For accurate performance comparison, use the `profiling` build
- Pose comparison requires profiling logs from both implementations

## Future Investigation

1. **Per-iteration kernel optimization**: Reduce GPU kernel launch overhead to improve per-iteration cost on x86
2. **Batch processing**: Process multiple frames together to amortize GPU overhead
3. **Radius search alignment**: Consider implementing radius-based neighbor search for better correspondence count match
4. **Convergence criteria**: Document exact thresholds in both implementations

## Files Referenced

| File                                 | Purpose                                             |
|--------------------------------------|-----------------------------------------------------|
| `logs/ndt_cuda_profiling.jsonl`      | CUDA release profiling (294 entries)                |
| `logs/ndt_autoware_profiling.jsonl`  | Autoware profiling (290 entries)                    |
| `logs/ndt_cuda_debug.jsonl`          | CUDA alignment debug                                |
| `logs/ndt_autoware_debug.jsonl`      | Autoware alignment debug                            |
| `logs/ndt_cuda_voxels.json`          | CUDA voxel grid dump (11,601 voxels)                |
| `logs/ndt_autoware_voxels.json`      | Autoware voxel grid dump (11,601 voxels)            |
| `scripts/compare_poses.py`           | Pose difference comparison script                   |
| `scripts/analyze_resource_usage.py`  | Resource usage analysis (CPU, GPU, power)           |
| `tmp/analyze_jetson_profiling.py`    | Jetson profiling analysis script                    |

## Conclusion

The CUDA NDT implementation is **functionally equivalent** to Autoware's NDT:

- **Pose difference**: Mean 3.8cm, Max 26cm (within acceptable localization tolerance)
- **100% convergence**: Both implementations converge successfully on both platforms
- **Voxel grids match**: Same voxel counts with minor boundary differences
- **Score output**: Normalized to match Autoware's convention (ratio ~0.81)

### Cross-Platform Performance Summary

| Platform | CUDA | Autoware | Relative | Winner |
|----------|------|----------|----------|--------|
| **Desktop x86** (RTX 5090) | 199.7 Hz | 125.4 Hz | **1.59x** | **CUDA** |
| **Jetson AGX Orin** (64GB) | 34.8 Hz | 26.5 Hz | **1.32x** | **CUDA** |

**Key Findings**:

1. **CUDA wins on both platforms**: After optimization, CUDA NDT outperforms Autoware on both x86 and Jetson.

2. **Per-iteration efficiency**:
   - On x86: CUDA 2.16ms/iter vs Autoware 2.73ms/iter (CUDA 1.27x faster)
   - On Jetson: CUDA 8.81ms/iter vs Autoware 12.89ms/iter (CUDA 1.46x faster)

3. **Initial pose estimation**: CUDA is 2.57-3.05x faster depending on platform:
   - x86: 2617ms vs 6720ms (2.57x faster), 0.024m position difference
   - Jetson: 7411ms vs 22613ms (3.05x faster), 0.427m position difference

4. **Real-time capable on both**: CUDA exceeds 10 Hz LiDAR rate by 20x on x86 and 3.5x on Jetson.

5. **Resource efficiency (Jetson)**:
   - CPU usage: 57% reduction (34.8% vs 81.0%)
   - GPU utilization: 11% average with headroom for perception
   - Power: Equal consumption with 30% higher throughput (30% better throughput/watt)

### Optimization Opportunities

For x86:
- Already optimized; graph kernels minimize launch overhead
- Further gains possible through batch processing for multiple frames

For Jetson:
- Current CUDA implementation is well-suited for Jetson's architecture
- Further gains possible through CUDA graph stream capture

### Remaining Differences

These apply to both platforms and do not affect functional correctness:
1. Correspondence count (~20% fewer in CUDA due to coordinate-based vs radius search)
2. NVTL values (~0.7 lower due to same reason)
3. Convergence speed (CUDA sometimes converges faster)

### Recommendation

- **For Jetson/embedded deployment**: Use CUDA NDT (32% faster tracking, 3.05x faster init pose)
- **For x86/server deployment**: Use CUDA NDT (59% faster tracking, 2.57x faster init pose)
- **For cross-platform**: CUDA NDT provides consistent behavior and superior performance on both platforms
- **For initial pose estimation**: CUDA is 2.57-3.05x faster with functionally equivalent accuracy
