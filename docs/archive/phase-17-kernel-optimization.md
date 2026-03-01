# Phase 17: Persistent Kernel with Cooperative Groups

## Overview

This phase eliminates per-iteration CPU-GPU transfers by moving the entire Newton optimization loop into a single persistent CUDA kernel using cooperative groups for grid synchronization.

**Current state:**
- ~10-14 kernel launches per iteration
- ~248 bytes CPU-GPU transfer per iteration
- 2.17x slower than Autoware's OpenMP implementation

**Target:**
- 1 kernel launch per alignment
- 0 bytes transfer during iteration
- <1.8x slower than Autoware

---

## Current Pipeline Analysis

### Per-Iteration Kernel Launches (10-14)

```
1. transform_points_kernel        - source_points × transform → transformed_points
2. spatial_hash_lookup_kernel     - transformed_points × hash_table → neighbors
3. compute_sin_cos_kernel         - pose → sin_cos [6]
4. compute_jacobians_kernel       - source_points × sin_cos → jacobians [N×18]
5. compute_point_hessians_kernel  - source_points × sin_cos → point_hessians [N×144]
6. compute_ndt_score_kernel       - transformed × neighbors × voxels → scores [N]
7. compute_ndt_gradient_kernel    - scores × jacobians → gradients [N×6]
8. compute_ndt_hessian_kernel_v2  - gradients × point_hessians → hessians [N×36]
9. CUB segmented_reduce           - 43 segments → reduce_output [43]
10. (cuSOLVER Newton solve)       - downloads H/g, solves, uploads delta
11-14. (Line search if enabled)
15. update_pose_kernel            - pose + delta × alpha → pose
16. check_convergence_kernel      - delta → converged_flag
```

### Per-Iteration CPU-GPU Transfers (~248 bytes)

| Transfer            | Direction | Size      | Purpose              |
|---------------------|-----------|-----------|----------------------|
| reduce_output       | GPU→CPU   | 172 bytes | Newton solve input   |
| delta               | CPU→GPU   | 24 bytes  | Newton step          |
| pose                | GPU→CPU   | 24 bytes  | Oscillation tracking |
| convergence flag    | GPU→CPU   | 4 bytes   | Loop control         |
| alpha (line search) | GPU→CPU   | 4 bytes   | Step length          |

---

## Persistent Kernel Design

### Data Location Strategy

**Registers (per-thread or solver-thread only):**
| Data            | Size                 | Notes                  |
|-----------------|----------------------|------------------------|
| pose            | 6 × f32 = 24 bytes   | Updated each iteration |
| delta           | 6 × f32 = 24 bytes   | Newton step            |
| sin_cos         | 6 × f32 = 24 bytes   | Recomputed inline      |
| transform       | 16 × f32 = 64 bytes  | Computed from pose     |
| H (solver only) | 36 × f64 = 288 bytes | Cholesky factorization |
| g (solver only) | 6 × f64 = 48 bytes   | Gradient for solve     |

**Shared Memory (per block):**
| Data           | Size                  | Notes                    |
|----------------|-----------------------|--------------------------|
| partial_sums   | 28 × blockDim.x × f32 | Reduction scratch        |
| converged_flag | 4 bytes               | Broadcast to all threads |

**Global Memory (read-only):**
| Data           | Size                | Notes               |
|----------------|---------------------|---------------------|
| source_points  | N × 3 × f32         | Input point cloud   |
| voxel_means    | V × 3 × f32         | Voxel grid          |
| voxel_inv_covs | V × 9 × f32         | Inverse covariances |
| hash_table     | capacity × 16 bytes | Spatial hash        |

**Global Memory (scratch):**
| Data          | Size     | Notes                       |
|---------------|----------|-----------------------------|
| reduce_buffer | 28 × f32 | Grid-level reduction output |

### Control Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ persistent_ndt_kernel (launched once with cudaLaunchCooperative)│
│                                                                 │
│ for iter = 0 to max_iterations:                                 │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ PHASE A: Per-point computation (N threads parallel)     │   │
│   │ - Compute sin_cos, transform from pose (inline)         │   │
│   │ - Transform point, hash query neighbors                 │   │
│   │ - Compute Jacobian + point Hessian (inline)             │   │
│   │ - Accumulate score + gradient + Hessian from neighbors  │   │
│   └─────────────────────────────────────────────────────────┘   │
│   grid.sync()                                                   │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ PHASE B: Block reduction (shared memory)                │   │
│   │ - Tree reduce 28 values within each block               │   │
│   │ - atomicAdd block results to global reduce_buffer       │   │
│   └─────────────────────────────────────────────────────────┘   │
│   grid.sync()                                                   │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ PHASE C: Newton solve (thread 0 of block 0 only)        │   │
│   │ - Load reduced gradient and Hessian                     │   │
│   │ - 6x6 Cholesky factorization in f64 registers           │   │
│   │ - Solve for delta, update pose                          │   │
│   │ - Check convergence, clear reduce_buffer                │   │
│   └─────────────────────────────────────────────────────────┘   │
│   grid.sync()                                                   │
│                                                                 │
│   if (converged) break;  // Uniform branch - no divergence      │
│                                                                 │
│ Write final pose, score, Hessian to output                      │
└─────────────────────────────────────────────────────────────────┘
```

### 6x6 Cholesky Solve in Registers

```cuda
__device__ void cholesky_solve_6x6_f64(
    double H[36],   // 6x6 Hessian (destroyed during factorization)
    double g[6],    // gradient in, solution out
    bool* success
) {
    // 1. Cholesky factorization: H = L * L^T
    for (int j = 0; j < 6; j++) {
        double sum = H[j*6 + j];
        for (int k = 0; k < j; k++)
            sum -= H[j*6 + k] * H[j*6 + k];
        if (sum <= 0.0) { *success = false; return; }
        H[j*6 + j] = sqrt(sum);
        for (int i = j+1; i < 6; i++) {
            sum = H[i*6 + j];
            for (int k = 0; k < j; k++)
                sum -= H[i*6 + k] * H[j*6 + k];
            H[i*6 + j] = sum / H[j*6 + j];
        }
    }

    // 2. Forward substitution: L * y = -g
    for (int i = 0; i < 6; i++) {
        double sum = -g[i];
        for (int j = 0; j < i; j++)
            sum -= H[i*6 + j] * g[j];
        g[i] = sum / H[i*6 + i];
    }

    // 3. Backward substitution: L^T * x = y
    for (int i = 5; i >= 0; i--) {
        double sum = g[i];
        for (int j = i+1; j < 6; j++)
            sum -= H[j*6 + i] * g[j];
        g[i] = sum / H[i*6 + i];
    }
    *success = true;
}
```

---

## Implementation Status

### Completed

| Phase | Description | Status |
|-------|-------------|--------|
| 17.1 | Device Functions | ✅ Complete |
| 17.2 | Persistent Kernel | ✅ Complete |
| 17.3 | FFI and Integration | ✅ Complete |
| 17.4 | Testing | ✅ Complete |
| 17.5 | NdtScanMatcher Integration | ✅ Complete |

**All Tests Passing** (2026-01-15):
- 5 persistent kernel tests pass
- 11 pipeline_v2 tests pass
- 3 align_full_gpu tests pass

### Files Created

```
src/cuda_ffi/
├── csrc/
│   ├── persistent_ndt.cu          # ✅ Persistent kernel with cooperative groups
│   ├── persistent_ndt_device.cuh  # ✅ Device functions (transforms, Jacobians, Hessians)
│   └── cholesky_6x6.cuh           # ✅ Register-based 6x6 Cholesky solver (f64)
├── src/
│   ├── persistent_ndt.rs          # ✅ Rust FFI wrapper
│   └── lib.rs                     # ✅ Export new bindings
└── build.rs                       # ✅ Compile new files

src/ndt_cuda/src/optimization/
└── full_gpu_pipeline_v2.rs        # ✅ Added can_use_persistent() and optimize_persistent()
```

### Phase 17.1: Device Functions ✅

Ported CubeCL kernels to CUDA C device functions in `persistent_ndt_device.cuh`:

- [x] `compute_sincos_inline()` - 6 trig values from pose angles
- [x] `compute_transform_inline()` - 4x4 transform matrix
- [x] `transform_point_inline()` - Apply transform to point
- [x] `compute_jacobians_inline()` - 3x6 Jacobian (18 values)
- [x] `compute_point_hessians_inline()` - Sparse Hessian (15 values)
- [x] `compute_ndt_contribution()` - Fused score/gradient/Hessian for all neighbors

### Phase 17.2: Persistent Kernel ✅

Implemented `persistent_ndt.cu`:

- [x] Per-point computation with fused score/gradient/Hessian
- [x] Block-level tree reduction in shared memory
- [x] Grid-level reduction with atomicAdd
- [x] Single-thread Newton solve with Cholesky (f64 precision)
- [x] Convergence check and pose update
- [x] Grid synchronization via `cooperative_groups::this_grid().sync()`

### Phase 17.3: FFI and Integration ✅

- [x] Created `persistent_ndt.rs` with safe Rust wrapper
- [x] Handle `cudaLaunchCooperativeKernel` requirements
- [x] Query grid size limits via `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
- [x] Added `can_use_persistent()` and `optimize_persistent()` to `FullGpuPipelineV2`
- [x] Legacy pipeline remains as fallback for large point clouds or line search

### Phase 17.4: Testing ✅

- [x] 4 persistent kernel FFI tests pass
- [x] 4 pipeline integration tests pass
- [x] Verified convergence matches legacy pipeline (scores within 2x)
- [x] Max cooperative blocks: 340 (supports ~87,000 points)

### Phase 17.5: NdtScanMatcher Integration ✅

Connected persistent kernel to main `NdtScanMatcher::align()`:

- [x] Auto-select persistent kernel when eligible (no config option needed)
- [x] Eligibility check via `can_use_persistent()`:
  - Point count >= 256 (minimum threshold for persistent kernel)
  - Line search disabled
  - Regularization disabled
  - Cooperative launch supported
  - Grid size fits
- [x] Fallback to legacy pipeline for small point counts, line search, or regularization
- [x] Integration in `align_full_gpu()` and `align_batch_gpu()`
- [ ] Benchmark with real rosbag data

### Bug Fix: Shared Memory Scope

Fixed critical bug where state variables (`pose`, `delta`, `converged`) were declared as `__shared__` (per-block) instead of using global memory (cross-block).

**Before (broken):**
```cuda
__shared__ float pose[6];        // Each block has own copy!
__shared__ uint32_t converged;   // Other blocks can't see block 0's update
```

**After (fixed):**
```cuda
// Use reduce_buffer for cross-block shared state:
// [0..27]   = reduction values
// [28]      = converged flag
// [29..34]  = current pose [6]
// [35..40]  = delta [6]
// [41]      = final score
float* g_pose = &reduce_buffer[29];
float* g_converged = &reduce_buffer[28];
```

### Remaining Work (Future)

1. **Correspondence count** - Persistent kernel doesn't output correspondence count
2. **Oscillation tracking** - Pose history not available during kernel execution
3. **Line search support** - Would require significant kernel restructuring
4. **Debug output** - Per-iteration debug data not available from persistent kernel

---

## Device Functions Reference

### Trigonometric (from `gpu_jacobian.rs`)

```cuda
__device__ __forceinline__ void compute_sincos_inline(
    const float pose[6],
    float* sr, float* cr, float* sp, float* cp, float* sy, float* cy
) {
    *sr = sinf(pose[3]); *cr = cosf(pose[3]);  // roll
    *sp = sinf(pose[4]); *cp = cosf(pose[4]);  // pitch
    *sy = sinf(pose[5]); *cy = cosf(pose[5]);  // yaw
}
```

### Transform Matrix

```cuda
__device__ __forceinline__ void compute_transform_inline(
    const float pose[6],
    float sr, float cr, float sp, float cp, float sy, float cy,
    float T[16]
) {
    T[0] = cy*cp;  T[1] = cy*sp*sr - sy*cr;  T[2] = cy*sp*cr + sy*sr;  T[3] = pose[0];
    T[4] = sy*cp;  T[5] = sy*sp*sr + cy*cr;  T[6] = sy*sp*cr - cy*sr;  T[7] = pose[1];
    T[8] = -sp;    T[9] = cp*sr;             T[10] = cp*cr;            T[11] = pose[2];
    T[12] = 0;     T[13] = 0;                T[14] = 0;                T[15] = 1;
}
```

### Jacobian (18 values per point)

```cuda
__device__ __forceinline__ void compute_jacobians_inline(
    float x, float y, float z,
    float sr, float cr, float sp, float cp, float sy, float cy,
    float J[18]
) {
    // dT/d(tx,ty,tz) = I
    J[0]=1; J[1]=0; J[2]=0; J[3]=0; J[4]=1; J[5]=0; J[6]=0; J[7]=0; J[8]=1;
    // dT/d(roll)
    J[9]  = (cy*sp*cr + sy*sr)*y + (-cy*sp*sr + sy*cr)*z;
    J[10] = (sy*sp*cr - cy*sr)*y + (-sy*sp*sr - cy*cr)*z;
    J[11] = cp*cr*y - cp*sr*z;
    // dT/d(pitch)
    J[12] = -cy*sp*x + cy*cp*sr*y + cy*cp*cr*z;
    J[13] = -sy*sp*x + sy*cp*sr*y + sy*cp*cr*z;
    J[14] = -cp*x - sp*sr*y - sp*cr*z;
    // dT/d(yaw)
    J[15] = -sy*cp*x + (-sy*sp*sr - cy*cr)*y + (-sy*sp*cr + cy*sr)*z;
    J[16] = cy*cp*x + (cy*sp*sr - sy*cr)*y + (cy*sp*cr + sy*sr)*z;
    J[17] = 0;
}
```

### Point Hessians (15 sparse values)

```cuda
__device__ __forceinline__ void compute_point_hessians_inline(
    float x, float y, float z,
    float sr, float cr, float sp, float cp, float sy, float cy,
    float pH[15]  // a2,a3,b2,b3,c2,c3,d1,d2,d3,e1,e2,e3,f1,f2,f3
) {
    pH[0] = (-cy*sp*sr + sy*cr)*y + (-cy*sp*cr - sy*sr)*z;  // a2
    pH[1] = (-sy*sp*sr - cy*cr)*y + (-sy*sp*cr + cy*sr)*z;  // a3
    pH[2] = -cp*sr*y - cp*cr*z;                              // b2
    pH[3] = -cy*cp*x - cy*sp*sr*y - cy*sp*cr*z;             // b3
    pH[4] = -sy*cp*x - sy*sp*sr*y - sy*sp*cr*z;             // c2
    pH[5] = sp*x - cp*sr*y - cp*cr*z;                        // c3
    pH[6] = -cy*cp*x + (-cy*sp*sr + sy*cr)*y + (-cy*sp*cr - sy*sr)*z;  // d1
    pH[7] = -sy*cp*x + (-sy*sp*sr - cy*cr)*y + (-sy*sp*cr + cy*sr)*z;  // d2
    pH[8] = 0;                                                // d3
    pH[9] = cy*cp*cr*y - cy*cp*sr*z;                         // e1
    pH[10] = sy*cp*cr*y - sy*cp*sr*z;                        // e2
    pH[11] = -sp*cr*y + sp*sr*z;                             // e3
    pH[12] = (-sy*sp*cr - cy*sr)*y + (sy*sp*sr - cy*cr)*z;   // f1
    pH[13] = (cy*sp*cr - sy*sr)*y + (-cy*sp*sr - sy*cr)*z;   // f2
    pH[14] = 0;                                               // f3
}
```

---

## Cooperative Kernel Launch

### Grid Size Calculation

```cuda
int numBlocksPerSm;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocksPerSm, persistent_ndt_kernel, BLOCK_SIZE, sharedMemBytes);

cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
int maxBlocks = prop.multiProcessorCount * numBlocksPerSm;

int numBlocks = (numPoints + BLOCK_SIZE - 1) / BLOCK_SIZE;
if (numBlocks > maxBlocks) {
    return use_legacy_pipeline();  // Fallback
}
```

### Launch

```cuda
size_t sharedMemBytes = BLOCK_SIZE * 28 * sizeof(float);
void* args[] = { /* kernel parameters */ };

cudaLaunchCooperativeKernel(
    (void*)persistent_ndt_kernel,
    dim3(numBlocks), dim3(BLOCK_SIZE),
    args, sharedMemBytes
);
```

---

## Expected Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Kernel launches | ~50/alignment | 1 | 98% reduction |
| CPU-GPU transfer | ~1KB/alignment | ~100 bytes | 90% reduction |
| Estimated time | 5.05ms | ~4.1ms | 19% faster |
| Ratio vs Autoware | 2.17x | ~1.76x | 19% closer |

---

## Success Criteria

- [x] All tests pass (349 tests, including 4 new persistent kernel tests)
- [x] Convergence rate matches legacy pipeline (verified in comparison test)
- [ ] Mean execution time < 4.5ms (pending benchmark with real data)
- [ ] Performance ratio < 1.9x vs Autoware (pending benchmark)
- [x] Zero CPU-GPU transfer during iteration loop
- [x] Single kernel launch per alignment
