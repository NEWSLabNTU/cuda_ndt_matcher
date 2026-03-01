# Phase 28: Legacy CubeCL Kernel Cleanup

## Background

The Phase 24 CUDA graph kernels (K1–K5 in `cuda_ffi/csrc/ndt_graph_kernels.cu`) replaced the earlier CubeCL-based optimization pipeline. Several CubeCL kernel files and their supporting infrastructure remain in the codebase despite being unreachable from the production pipeline. This phase removes them.

## Inventory

### Fully dead — no production callers

| File | Lines | Contents | Superseded by |
|------|-------|----------|---------------|
| `optimization/gpu_pipeline_kernels.rs` | 2023 | 10 CubeCL kernels (transform, dot product, candidates, batch transform/score, More-Thuente, convergence, regularization, pose update) + 13 tests | Graph K1–K5 |
| `derivatives/gpu_jacobian.rs` | 553 | `compute_sin_cos_kernel`, `compute_jacobians_kernel`, `compute_point_hessians_kernel` + tests | Graph K2 (inline Jacobians) |
| `voxel_grid/gpu_builder.rs` | 860 | `GpuVoxelGridBuilder` — CubeCL voxel construction with Morton codes + 9 tests | cuda_ffi hash table pipeline |
| `voxel_grid/gpu/morton.rs` | 388 | `compute_morton_codes_kernel`, `pack_morton_codes_kernel` | cuda_ffi hash table |

**Total: 3824 lines**

### Partially dead — scoring methods still used

| File | Lines | Status |
|------|-------|--------|
| `runtime.rs` | 1401 | `compute_scores` and `compute_nvtl_scores` are called from `ndt.rs` for NVTL/TP scoring. The remaining 5 methods (`compute_derivatives`, `compute_derivatives_with_metric`, `transform_points`) are dead — only called from tests. |
| `derivatives/gpu.rs` | 3407 | `radius_search_kernel`, `compute_ndt_score_kernel`, `compute_ndt_nvtl_kernel` are used by `runtime.rs` scoring methods. The remaining kernels are dead: `compute_ndt_gradient_kernel`, `compute_ndt_gradient_point_to_plane_kernel`, `compute_ndt_hessian_kernel`, `compute_ndt_hessian_kernel_v2`, `compute_ndt_score_point_to_plane_kernel`. |

## Work Items

### 28.1 Delete fully-dead files

Delete these files and remove their `mod`/`pub use` declarations:

- [ ] `src/ndt_cuda/src/optimization/gpu_pipeline_kernels.rs`
  - Remove `pub mod gpu_pipeline_kernels` from `optimization/mod.rs`
  - Remove `pub use gpu_pipeline_kernels::{...}` from `optimization/mod.rs`
  - Move `DEFAULT_NUM_CANDIDATES` constant to `full_gpu_pipeline_v2.rs` (only remaining user)
- [ ] `src/ndt_cuda/src/derivatives/gpu_jacobian.rs`
  - Remove `pub mod gpu_jacobian` from `derivatives/mod.rs`
  - Remove re-exports from `derivatives/mod.rs`
- [ ] `src/ndt_cuda/src/voxel_grid/gpu_builder.rs`
  - Remove `pub mod gpu_builder` from `voxel_grid/mod.rs`
  - Remove `pub use gpu_builder::GpuVoxelGridBuilder` from `voxel_grid/mod.rs`
  - Remove `pub use voxel_grid::GpuVoxelGridBuilder` from `lib.rs`
  - Remove `VoxelGrid::from_points_gpu` method (no callers outside tests)
- [ ] `src/ndt_cuda/src/voxel_grid/gpu/morton.rs`
  - Remove `pub mod morton` from `voxel_grid/gpu/mod.rs`

### 28.2 Prune dead kernels from `derivatives/gpu.rs`

Keep the 3 kernels used by `runtime.rs` scoring:
- `radius_search_kernel`
- `compute_ndt_score_kernel`
- `compute_ndt_nvtl_kernel`
- `transform_point_inline` (helper used by the above)

Delete the 5 unused kernels and their tests:
- [ ] `compute_ndt_gradient_kernel`
- [ ] `compute_ndt_gradient_point_to_plane_kernel`
- [ ] `compute_ndt_hessian_kernel`
- [ ] `compute_ndt_hessian_kernel_v2`
- [ ] `compute_ndt_score_point_to_plane_kernel`

Update `derivatives/mod.rs` re-exports to remove deleted symbols.

### 28.3 Prune dead methods from `runtime.rs`

Keep:
- `GpuRuntime::new`, `with_device_id`, `is_cuda_available`
- `compute_scores`
- `compute_nvtl_scores`

Delete (only used by tests within `runtime.rs`):
- [ ] `compute_derivatives` (+ its tests)
- [ ] `compute_derivatives_with_metric` (+ its tests)
- [ ] `transform_points` (+ its tests)

Remove dead imports (`compute_ndt_gradient_*`, `compute_ndt_hessian_*`, `compute_ndt_score_point_to_plane_*`, `compute_point_hessians_cpu`, `compute_point_jacobians_cpu`).

### 28.4 Clean up exports

- [ ] `lib.rs`: Remove `GpuVoxelGridBuilder` and `GpuDerivativeResult` exports
- [ ] `derivatives/mod.rs`: Remove re-exports of deleted symbols (`compute_ndt_hessian_kernel_v2`, `compute_point_hessians_cpu`, `compute_point_jacobians_cpu`, etc.)
- [ ] `optimization/mod.rs`: Remove re-exports of deleted `gpu_pipeline_kernels` symbols
- [ ] Audit remaining `pub use` statements for any newly-orphaned exports

### 28.5 Verify

- [ ] `cargo build --features cuda` succeeds
- [ ] `cargo test --features cuda` — all remaining tests pass
- [ ] `cargo clippy` reports no new warnings
- [ ] No `#[allow(dead_code)]` annotations were added to suppress warnings from this cleanup

## Estimated Reduction

3436 lines deleted outright (28.1), 3817 lines pruned (28.2–28.3). Net removal: **7253 lines**.

Note: `voxel_grid/gpu/morton.rs` was kept (388 lines) — its CubeCL kernels are used by the GPU voxel pipeline (`pipeline.rs`), not the optimization pipeline.

## Risks

- **`runtime.rs` scoring path**: The NVTL/TP scoring still uses CubeCL kernels via `GpuRuntime`. These must be kept (or migrated to cuda_ffi in a future phase). This cleanup only removes the derivative/optimization methods that are dead.
- **Test coverage**: Deleted files contain ~25 tests. These test kernels that are no longer called, so removing them is safe. The equivalent functionality is tested through the graph kernel integration tests.
