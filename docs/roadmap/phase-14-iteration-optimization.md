# Phase 14: Per-Iteration GPU Optimization

**Status**: 🔲 Planned
**Priority**: High
**Target**: Reduce per-iteration memory transfer from ~490 KB to ~64 bytes

## Overview

The current GPU derivative pipeline uploads Jacobians and point Hessians from CPU every iteration. This phase moves these computations to GPU and optimizes buffer management.

## Current State

### Per-Iteration Transfer Analysis

From `derivatives/pipeline.rs:461-599`, each iteration transfers:

| Direction | Data | Size (N=756 points) | Notes |
|-----------|------|---------------------|-------|
| CPU → GPU | Transform matrix | 64 bytes | Required |
| CPU → GPU | Jacobians | N × 18 × 4 = 54 KB | **Can be GPU** |
| CPU → GPU | Point Hessians | N × 144 × 4 = 435 KB | **Can be GPU** |
| GPU → CPU | Reduced results | 43 × 4 = 172 bytes | Required |

**Total upload per iteration**: ~490 KB
**For 15-30 iterations**: 7-15 MB per alignment

### Code Location

```rust
// derivatives/pipeline.rs:478-489
// Compute Jacobians and point Hessians on CPU using cached source points
let jacobians = compute_point_jacobians_cpu(&self.cached_source_points, pose);
let point_hessians = compute_point_hessians_cpu(&self.cached_source_points, pose);

// Upload Jacobians - NEW ALLOCATION EVERY ITERATION
self.jacobians = self.client.create(f32::as_bytes(&jacobians));

// Combine jacobians and point_hessians for Hessian kernel
let mut jacobians_combined = jacobians.clone();
jacobians_combined.extend_from_slice(&point_hessians);
self.jacobians_combined = self.client.create(f32::as_bytes(&jacobians_combined));
```

## Work Items

### P3: GPU Jacobian/Point Hessian Kernels

**Goal**: Compute Jacobians and point Hessians directly on GPU.

**Input data** (already on GPU):
- Source points [N × 3] - cached in `self.source_points`
- Pose [6 floats] - can be extracted from transform matrix or uploaded separately

**Jacobian formula** (from `derivatives/jacobian.rs`):

The Jacobian ∂T(x)/∂p for each point depends on:
- Point coordinates (x, y, z)
- Pose angles (roll, pitch, yaw)

```
j_ang terms (8 rotation derivatives):
  j_ang_a = -sx*sz + cx*sy*cz    j_ang_b = -sx*cz - cx*sy*sz
  j_ang_c =  cx*sz + sx*sy*cz    j_ang_d =  cx*cz - sx*sy*sz
  j_ang_e = -sy*cz               j_ang_f =  cy*sz
  j_ang_g =  cy*cz               j_ang_h =  sy*sz

Full 3x6 Jacobian per point:
  [[1, 0, 0, 0,           cy*sz*z+cy*cz*y, ...],
   [0, 1, 0, ...                          ],
   [0, 0, 1, ...                          ]]
```

**Point Hessian formula** (from `derivatives/point_hessian.rs`):

Second derivatives ∂²T(x)/∂p² with 15 unique terms (h_ang_a2, h_ang_a3, h_ang_b2, b3, c2, c3, d1, d2, d3, e1, e2, e3, f1, f2, f3).

**Implementation steps**:

1. Add `compute_jacobians_kernel` to `derivatives/gpu.rs`:
   ```rust
   #[cube(launch_unchecked)]
   fn compute_jacobians_kernel(
       source_points: &Array<f32>,      // [N × 3]
       sin_cos: &Array<f32>,            // [6]: sin(r), cos(r), sin(p), cos(p), sin(y), cos(y)
       num_points: u32,
       jacobians_out: &mut Array<f32>,  // [N × 18]
   ) {
       // Compute j_ang terms from sin_cos
       // Compute 3×6 Jacobian per point
   }
   ```

2. Add `compute_point_hessians_kernel` to `derivatives/gpu.rs`:
   ```rust
   #[cube(launch_unchecked)]
   fn compute_point_hessians_kernel(
       source_points: &Array<f32>,      // [N × 3]
       sin_cos: &Array<f32>,            // [6]
       num_points: u32,
       point_hessians_out: &mut Array<f32>,  // [N × 144]
   ) {
       // Compute h_ang terms from sin_cos
       // Compute 15 second derivative matrices per point
   }
   ```

3. Modify `compute_iteration_gpu_reduce()`:
   - Upload sin/cos values (6 floats) instead of jacobians
   - Call GPU kernels before gradient/hessian computation
   - Remove CPU jacobian computation

**Estimated effort**: ~200 LOC
**Expected impact**: 490 KB → 24 bytes upload per iteration (sin/cos values)

### P4: Pre-allocated Buffer Reuse

**Goal**: Eliminate per-iteration GPU memory allocations.

**Current issue** (pipeline.rs:484,489):
```rust
self.jacobians = self.client.create(...);           // NEW allocation
self.jacobians_combined = self.client.create(...);  // NEW allocation
```

**Solution**: Pre-allocate buffers at pipeline creation.

**Implementation steps**:

1. Add fields to `GpuDerivativePipeline`:
   ```rust
   pub struct GpuDerivativePipeline {
       // ... existing fields ...

       // Pre-allocated iteration buffers
       jacobians: Handle,           // [max_points × 18]
       point_hessians: Handle,      // [max_points × 144]
       jacobians_combined: Handle,  // [max_points × 162]
   }
   ```

2. Allocate in `new()`:
   ```rust
   let jacobians = client.empty(max_points * 18 * size_of::<f32>());
   let point_hessians = client.empty(max_points * 144 * size_of::<f32>());
   let jacobians_combined = client.empty(max_points * 162 * size_of::<f32>());
   ```

3. Use existing handles in `compute_iteration_gpu_reduce()`:
   - Write to pre-allocated buffers instead of creating new ones
   - If P3 is implemented, kernels write directly to these buffers

**Estimated effort**: ~50 LOC
**Expected impact**: Reduce allocation overhead, improve cache coherence

### P5: Full GPU Newton Loop (Future)

**Goal**: Run entire Newton optimization on GPU, download only final result.

**Challenges**:
- 6×6 matrix inversion on GPU (small matrix, may not benefit)
- Convergence checking requires conditional control flow
- CubeCL may not support required operations

**Approach**:
- Implement Newton step on GPU using LU decomposition or direct formula
- Use atomic flags for convergence status
- May require native CUDA kernel via `cuda_ffi`

**Estimated effort**: Complex, may not be worthwhile
**Expected impact**: Eliminate ~30 sync points per alignment

### P6: Scan Queue Batch Processing (Future)

**Goal**: Process multiple incoming scans together to amortize setup costs.

**Architecture**:
```
Incoming scans → Queue (capacity 2-3) → Batch processor → Results
```

**Trade-offs**:
- Pros: Amortize voxel upload, pipeline setup
- Cons: Adds latency (must wait for batch)

**Constraints**:
- Real-time requirement: 100ms total latency budget
- Typical scan rate: 10-20 Hz
- Queue fill time: 50-100ms for batch of 2

**Estimated effort**: Medium
**Expected impact**: Amortize ~5ms setup across multiple scans

## Implementation Order

1. **P4: Buffer reuse** (quick win, required for P3)
2. **P3: GPU Jacobians** (highest impact)
3. **P5/P6**: Future optimization based on profiling results

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Per-iteration upload | ~490 KB | <100 bytes |
| GPU allocations per iteration | 2-3 | 0 |
| Mean alignment time | 13.07 ms | <5 ms |

## Files to Modify

| File | Changes |
|------|---------|
| `derivatives/gpu.rs` | Add jacobian/hessian kernels |
| `derivatives/pipeline.rs` | Add pre-allocated buffers, call GPU kernels |
| `derivatives/mod.rs` | Export new kernels |

## Testing

1. **Unit tests**: Compare GPU vs CPU jacobian/hessian computation
2. **Integration test**: Verify `align_gpu()` produces same results
3. **Performance test**: Measure per-iteration timing breakdown
4. **Stress test**: Run on large point clouds (10k+ points)

## References

- `derivatives/jacobian.rs` - CPU Jacobian formulas
- `derivatives/point_hessian.rs` - CPU point Hessian formulas
- `derivatives/angular.rs` - j_ang and h_ang term definitions
- `docs/profiling-results.md` - Current performance measurements
