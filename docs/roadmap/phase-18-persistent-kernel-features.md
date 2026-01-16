# Phase 18: Persistent Kernel Full Features

This phase adds the remaining features to the persistent kernel to achieve full feature parity with the legacy multi-kernel pipeline.

## Current State

The persistent kernel (`persistent_ndt.cu`) runs the entire Newton optimization loop in a single CUDA kernel launch using cooperative groups for grid-wide synchronization. It supports:

- Basic Newton iteration (score, gradient, Hessian computation)
- Hash table voxel lookup (O(27) neighbors)
- Cholesky-based 6x6 linear solver
- Convergence check based on delta norm
- ✅ Final Hessian output for covariance estimation
- ✅ GNSS regularization (score, gradient, Hessian adjustments)
- ✅ Correspondence count tracking
- ✅ Oscillation detection (incremental, direction reversal tracking)
- ✅ Line search (Strong Wolfe conditions with K=8 candidates and early termination)

## Missing Features

### 18.1 Line Search Support (Medium Priority)

**Status:** ✅ Complete - Sequential evaluation approach (Option A)

The persistent kernel currently uses fixed step size (alpha=1.0). This phase adds More-Thuente line search with K=8 candidate step sizes.

**Implementation: Option A - Sequential K-Candidate Evaluation**

Process each candidate sequentially, reusing existing score/gradient computation:

1. **Add line search parameters to kernel signature:**
   - `int32_t ls_enabled` - Enable flag
   - `int32_t ls_num_candidates` - Number of candidates (default: 8)
   - `float ls_mu` - Armijo constant (default: 1e-4)
   - `float ls_nu` - Curvature constant (default: 0.9)

2. **Extend reduce_buffer layout** (56 → 96 floats):
   ```
   [56..63]  = phi_candidates[8]     - scores at each candidate
   [64..71]  = dphi_candidates[8]    - directional derivatives
   [72..79]  = alpha_candidates[8]   - step sizes
   [80..85]  = original_pose[6]      - saved pose before line search
   [86]      = phi_0                 - score at current pose
   [87]      = dphi_0                - directional derivative at current pose
   [88]      = best_alpha            - selected step size
   [89..95]  = reserved for alignment
   ```

3. **Line search phase after Newton solve:**
   ```
   PHASE B: Line Search (if ls_enabled)
     B1. Save original_pose = g_pose
     B2. Compute phi_0 = score, dphi_0 = gradient · delta
     B3. Generate K candidates (golden ratio decay: 1.0, 0.618, 0.382, ...)
     B4. For each candidate k (with early termination):
         - Set g_pose = original_pose + alpha[k] * delta
         - grid.sync()
         - Compute score and gradient (reuse Phase A code, skip Hessian)
         - grid.sync()
         - Store phi[k], dphi[k]
         - If Wolfe conditions satisfied, early terminate
     B5. Apply More-Thuente selection for best_alpha
     B6. Set g_pose = original_pose + best_alpha * delta
   ```

4. **Optimizations:**
   - Reuse Jacobians from original pose (like legacy pipeline)
   - Skip Hessian computation during line search
   - Early termination when first candidate satisfies Wolfe conditions

**Performance Impact:**
| Scenario | grid.sync() calls | Relative time |
|----------|-------------------|---------------|
| No line search | 1 per iter | 1.0x |
| LS early term (k=0) | 5 per iter | ~1.5x |
| LS full (K=8) | 18 per iter | ~3.0x |
| LS average (k=3) | 9 per iter | ~2.0x |

**Complexity:** ~150 lines of CUDA code

### 18.2 GNSS Regularization Support (Medium Priority)

**Status:** ✅ Complete

Regularization penalizes longitudinal drift from a GNSS reference position. This is important for open road scenarios where NDT can drift along the road direction.

**Required Changes:**

1. Add regularization parameters to kernel signature:
   - `float reg_ref_x, reg_ref_y` (GNSS reference position)
   - `float reg_scale` (regularization weight)
   - `int reg_enabled` (enable flag)

2. Add regularization computation after reduction:
   ```cuda
   if (reg_enabled && threadIdx.x == 0 && blockIdx.x == 0) {
       float dx = reg_ref_x - g_pose[0];
       float dy = reg_ref_y - g_pose[1];
       float yaw = g_pose[5];
       float sin_yaw = sinf(yaw);
       float cos_yaw = cosf(yaw);

       // Longitudinal distance in vehicle frame
       float longitudinal = dy * sin_yaw + dx * cos_yaw;

       // Add to score, gradient, Hessian
       // (see apply_regularization_kernel in gpu_pipeline_kernels.rs)
   }
   ```

**Complexity:** Low (single-thread addition, ~50 lines)

### 18.3 Correspondence Count Tracking (Low Priority)

**Status:** ✅ Complete

The legacy pipeline tracks the number of point-voxel correspondences for diagnostics and quality assessment.

**Required Changes:**

1. Add output parameter: `uint32_t* out_num_correspondences`

2. Track correspondences during reduction:
   ```cuda
   // In per-point computation
   int my_correspondences = num_neighbors;

   // Add to block reduction
   // Add correspondence_sum segment to reduce_buffer
   ```

3. Output final correspondence count

**Complexity:** Low (~30 lines)

### 18.4 Oscillation Detection (Low Priority)

**Status:** ✅ Complete

Oscillation detection tracks when the optimizer reverses direction, indicating potential instability.

**Implementation:**

Instead of storing full pose history, the kernel computes oscillation count incrementally:
- Track `prev_prev_pos` and `prev_pos` in reduce_buffer (6 floats total)
- After each pose update, compute movement vectors and their cosine
- Cosine < -0.9 (about 154 degrees) indicates direction reversal
- Count consecutive reversals, track max across all iterations
- Output `out_max_oscillation_count` at the end

**Buffer Layout Update:**
- `reduce_buffer[44..46]` = prev_prev_pos [3]
- `reduce_buffer[47..49]` = prev_pos [3]
- `reduce_buffer[50]` = current oscillation count
- `reduce_buffer[51]` = max oscillation count

**Complexity:** Low (~60 lines, minimal memory overhead)

### 18.5 Final Hessian Output (Low Priority)

**Status:** ✅ Complete

The final Hessian is needed for covariance estimation. Currently, the reduce buffer is cleared before the final iteration completes, losing the Hessian.

**Required Changes:**

1. Don't clear reduce_buffer on final iteration:
   ```cuda
   if (iter < max_iterations - 1 && *g_converged < 0.5f) {
       // Clear only if continuing
       for (int i = 0; i < 28; i++) {
           reduce_buffer[i] = 0.0f;
       }
   }
   ```

2. Expand upper-triangular Hessian to full 6×6 matrix in output

**Complexity:** Very Low (~20 lines)

## Implementation Order

Recommended priority based on impact vs. complexity:

1. ✅ **18.5 Final Hessian** - Quick fix, enables covariance estimation
2. ✅ **18.2 Regularization** - Low complexity, important for GNSS areas
3. ✅ **18.3 Correspondence Count** - Low complexity, useful for diagnostics
4. ✅ **18.4 Oscillation Detection** - Incremental tracking, useful for diagnostics
5. ✅ **18.1 Line Search** - Sequential evaluation with early termination (Option A)

## Testing Strategy

1. **Unit Tests:**
   - `test_persistent_with_regularization` - Verify regularization effect ✅
   - `test_persistent_correspondence_count` - Verify count matches legacy ✅
   - `test_persistent_hessian_output` - Verify Hessian is non-zero ✅
   - `test_persistent_oscillation_count` - Verify oscillation detection ✅

2. **Integration Tests:**
   - Run rosbag with persistent kernel + all features
   - Compare pose trajectory with legacy pipeline
   - Verify convergence rate and iteration count

3. **Line Search Coverage:**
   - `test_persistent_kernel_eligible_with_line_search` - Verify persistent kernel is eligible with line search ✅
   - Persistent kernel now supports line search with Strong Wolfe conditions and early termination

## Memory/Performance Impact

| Feature | Additional Memory | Performance Impact |
|---------|-------------------|-------------------|
| Line Search | K × N × 3 × 4 bytes | +~200% per iteration |
| Regularization | 16 bytes params | Negligible |
| Correspondence Count | 4 bytes output | Negligible |
| Oscillation Detection | 32 bytes (8 floats) | Negligible |
| Final Hessian | 0 (already allocated) | Negligible |

Line search is the only feature with significant performance impact, as it requires K additional score evaluations per iteration. However, it may reduce total iterations needed for convergence.
