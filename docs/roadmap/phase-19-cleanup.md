# Phase 19: Cleanup & Enhancements

This phase covers cleanup tasks after removing the legacy multi-kernel pipeline, plus enhancements to restore debug features in the persistent kernel.

## Status

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| 19.1 Remove unused struct fields | Low | ✅ Complete | Removed 25+ legacy Handle fields |
| 19.2 Remove unused GpuNewtonSolver | Low | ✅ Complete | Removed from struct (kept module for tests) |
| 19.3 Per-iteration alpha tracking | Medium | ✅ Complete | Accumulates in reduce_buffer[86], returns avg_alpha |
| 19.4 Per-iteration debug data | Low | ✅ Complete | 50 floats/iter, zero overhead when disabled |

## 19.1 Remove Unused Struct Fields

**Priority:** Low (cosmetic, reduces warnings)

The following fields in `FullGpuPipelineV2` are allocated but never used after removing the legacy pipeline:

```rust
// Legacy pose/transform buffers
pose_gpu: Handle,           // [6]
delta_gpu: Handle,          // [6]
sin_cos: Handle,            // [6]
transform: Handle,          // [16]
best_alpha: Handle,         // [1]
ls_converged: Handle,       // [1]

// Legacy per-point computation buffers
jacobians: Handle,          // [N × 18]
point_hessians: Handle,     // [N × 144]
transformed_points: Handle, // [N × 3]
neighbor_indices: Handle,   // [N × MAX_NEIGHBORS]
neighbor_counts: Handle,    // [N]

// Legacy reduction buffers
scores: Handle,             // [N]
correspondences: Handle,    // [N] u32
correspondences_f32: Handle,// [N] f32
gradients: Handle,          // [N × 6]
hessians: Handle,           // [N × 36]
reduce_output: Handle,      // [43]
gradient_reduced: Handle,   // [6]

// Legacy line search buffers
candidates: Handle,         // [K]
batch_transformed: Handle,  // [K × N × 3]
batch_scores: Handle,       // [K × N]
batch_dir_derivs: Handle,   // [K × N]
phi_cache: Handle,          // [K]
dphi_cache: Handle,         // [K]
phi_0: Handle,              // [1]
dphi_0: Handle,             // [1]

// Legacy misc
converged_flag: Handle,     // [1] u32
reduce_temp: Handle,
reduce_temp_bytes: usize,
ls_params: Handle,          // [3]
correspondence_sum: Handle, // [1]
```

**Implementation:**
1. Remove fields from struct definition
2. Remove allocations from `new()` constructor
3. Update any remaining references

**Memory savings:** ~(N × 200 + K × N × 20) bytes per pipeline instance

## 19.2 Remove Unused GpuNewtonSolver

**Priority:** Low

The `newton_solver: GpuNewtonSolver` field is unused - the persistent kernel uses its own Cholesky solver. Remove:
- Field from `FullGpuPipelineV2`
- `GpuNewtonSolver` struct if no longer used elsewhere

## 19.3 Per-Iteration Alpha Tracking

**Priority:** Medium (useful for diagnostics)

**Status:** ✅ Complete

The persistent kernel now tracks accumulated step sizes for computing `avg_alpha`.

**Implementation:**

1. Added `g_alpha_sum` accumulator at `reduce_buffer[86]` in CUDA kernel
2. After each iteration's pose update, accumulates `best_alpha` (from line search) or `1.0` (no line search)
3. Added `out_alpha_sum` output parameter to kernel and FFI
4. `optimize()` computes `avg_alpha = alpha_sum / iterations`

**Behavior:**
- With line search enabled: `avg_alpha` reflects actual step sizes taken
- With line search disabled: `avg_alpha = 1.0` (full Newton step each iteration)
- When line search finds no improvement: `alpha = 0` (no step taken)

## 19.4 Per-Iteration Debug Data

**Priority:** Low (development/debugging only)

**Status:** ✅ Complete

See [phase-19-4-debug-buffer-design.md](phase-19-4-debug-buffer-design.md) for detailed design.

### Summary

The persistent kernel can collect per-iteration debug data with **zero overhead when disabled**:

- **Buffer layout:** 50 floats per iteration (score, pose, gradient, Hessian, delta, alpha, etc.)
- **Total size:** ~6 KB for 30 iterations
- **Guard:** Single `if (debug_enabled)` check per iteration
- **When disabled:** No allocation, no writes, no download

### Key Design Points

1. Add `debug_enabled: i32` and `debug_buffer: *mut f32` kernel parameters
2. Allocate buffer only when `config.enable_debug` is true
3. Pass nullptr (0) when disabled - no memory writes
4. Parse buffer into `Vec<IterationDebug>` on CPU side

## Implementation Order

1. **19.1 + 19.2** ✅ - Struct cleanup, warnings removed
2. **19.3** ✅ - Alpha tracking for diagnostics
3. **19.4** ✅ - Per-iteration debugging with zero overhead when disabled
