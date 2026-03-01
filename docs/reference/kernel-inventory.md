# GPU Kernel Inventory

Complete inventory of all CUDA and CubeCL GPU kernels in the project.

## CubeCL Kernels (Rust → CUDA)

Kernels compiled via CubeCL using the `#[cube(launch_unchecked)]` attribute.

### Filtering (`ndt_cuda/src/filtering/kernels.rs`)

| Kernel | Line | Description |
|--------|------|-------------|
| `compute_filter_mask_kernel` | 17 | Per-point distance/Z-height filter → mask (1=keep, 0=discard) |
| `compact_points_kernel` | 59 | Compact points using mask + prefix sum |
| `prefix_sum_kernel` | 92 | Placeholder (currently CPU-based) |
| `compute_voxel_centroids_kernel` | 109 | Placeholder (currently CPU-based) |

### Voxel Grid (`ndt_cuda/src/voxel_grid/kernels.rs`)

| Kernel | Line | Description |
|--------|------|-------------|
| `compute_voxel_ids_kernel` | 25 | Compute voxel IDs from point positions and grid parameters |
| `transform_points_kernel` | 76 | Transform N points by single 4×4 matrix |
| `transform_points_batch_kernel` | 128 | Transform N points by K poses simultaneously (2D grid: `(ceil(N/256), K)`) |
| `compute_transforms_batch_kernel` | 191 | Compute K 4×4 matrices from K Euler-angle poses |
| `compute_sin_cos_batch_kernel` | 272 | Precompute sin/cos for K poses |

### Morton Codes (`ndt_cuda/src/voxel_grid/gpu/morton.rs`)

| Kernel | Line | Description |
|--------|------|-------------|
| `compute_morton_codes_kernel` | 46 | Compute 63-bit Morton codes (Z-order) for N points; split into u32 pairs |
| `pack_morton_codes_kernel` | 110 | Pack split Morton codes into u64 format |

Helpers: `expand_bits_21` (L131), `morton_encode_split` (L158).

### Derivatives (`ndt_cuda/src/derivatives/gpu.rs`, `gpu_jacobian.rs`)

| Kernel | Line | File | Description |
|--------|------|------|-------------|
| `radius_search_kernel` | 61 | `gpu.rs` | Brute-force multi-voxel radius search (O(N×V)), up to 8 neighbors |
| `compute_sin_cos_kernel` | 21 | `gpu_jacobian.rs` | Compute sin/cos from pose angles |
| `compute_jacobians_kernel` | 50 | `gpu_jacobian.rs` | Per-point 3×6 Jacobians (transformation derivatives) |

### Scoring (`ndt_cuda/src/scoring/gpu.rs`)

| Kernel | Line | Description |
|--------|------|-------------|
| `compute_scores_batch_kernel` | 50 | Batched NDT scoring: M poses × N points, radius search + Gaussian score |

### GPU Pipeline / Line Search (`ndt_cuda/src/optimization/gpu_pipeline_kernels.rs`)

| Kernel | Line | Description |
|--------|------|-------------|
| `cast_u32_to_f32_kernel` | 33 | Type conversion for CUB reduce compatibility |
| `compute_transform_from_sincos_kernel` | 63 | Build 4×4 transform from sin/cos + translation (single-thread) |
| `dot_product_6_kernel` | 119 | 6-element dot product for directional derivative (single-thread) |
| `generate_candidates_kernel` | 144 | Generate 8 line search step-size candidates |
| `update_pose_kernel` | 210 | `pose += delta × alpha` |
| `check_convergence_kernel` | 232 | Check if delta norm < epsilon |
| `apply_regularization_kernel` | 287 | Add GNSS regularization to Hessian diagonal |
| `batch_transform_kernel` | 377 | Transform N points by multiple step-adjusted poses |
| `batch_score_gradient_kernel` | 459 | Evaluate score/gradient for line search candidates |
| `more_thuente_kernel` | 978 | More-Thuente line search: Strong Wolfe condition evaluation |

---

## Raw CUDA Kernels (`__global__`)

Traditional CUDA C++ kernels in `cuda_ffi/csrc/`.

### Graph Kernels — Phase 24 (`csrc/ndt_graph_kernels.cu`)

The primary NDT iteration pipeline, captured into a CUDA graph for low-latency replay.

| Kernel | Line | Grid | Description |
|--------|------|------|-------------|
| K1: `ndt_graph_init_kernel` | 113 | 1×1 | Initialize optimization state from initial pose |
| K2: `ndt_graph_compute_kernel` | 163 | ceil(N/256)×256 | Per-point score/gradient/Hessian + block reduction |
| K3: `ndt_graph_solve_kernel` | 311 | 1×32 | Newton solve: Cholesky factorization + backsubstitution |
| K4: `ndt_graph_linesearch_kernel` | 473 | variable | Parallel line search candidate evaluation |
| K5: `ndt_graph_update_kernel` | 618 | 1×1 | Apply step, check convergence/oscillation |

Supporting headers: `ndt_graph_common.cuh`, `cholesky_6x6.cuh`, `warp_reduce.cuh`, `warp_cholesky.cuh`.

### Voxel Hash Table (`csrc/voxel_hash.cu`)

| Kernel | Line | Grid | Description |
|--------|------|------|-------------|
| `voxel_hash_build_kernel` | 69 | ceil(V/256)×256 | Build spatial hash table from voxel means (linear probing) |
| `voxel_hash_query_kernel` | 136 | ceil(N/256)×256 | Query 27-neighbor cube for radius search |
| `count_entries_kernel` | 340 | — | Debug: count non-empty entries |

### Batch Persistent NDT (`csrc/batch_persistent_ndt.cu`) — Legacy

Replaced by graph kernels (Phase 24) for single alignments. Still used for M-way batch alignments (initial pose).

| Kernel | Line | Description |
|--------|------|-------------|
| `batch_persistent_ndt_kernel` | 259 | M parallel alignments with atomic barriers |
| `batch_persistent_ndt_kernel_textured` | 957 | Texture-cache variant for improved read bandwidth |
| `batch_persistent_ndt_kernel_warp_optimized` | 1633 | Warp-level reduction, reduced synchronization |

### Segment Detection (`csrc/segment_detect.cu`)

| Kernel | Line | Description |
|--------|------|-------------|
| `detect_boundaries_kernel` | 27 | Mark where sorted Morton codes change (voxel boundaries) |
| `iota_kernel` | 153 | Generate sequence 0..N-1 |
| `gather_u64_kernel` | 185 | Gather u64 values at specified indices |

---

## CUB / cuSOLVER Primitives (via FFI)

Library calls wrapped in `cuda_ffi/src/`.

| Module | Functions | Purpose |
|--------|-----------|---------|
| `radix_sort.rs` | `cub_radix_sort_pairs_u64_u32` | Sort (Morton code, point index) pairs |
| `segment_detect.rs` | `cub_detect_boundaries`, `cub_inclusive_sum_u32`, `cub_select_flagged_u32`, `cub_iota_u32` | Voxel segmentation |
| `segmented_reduce.rs` | `cub_segmented_reduce_sum_f32/f64` | Per-voxel reduction (mean, covariance sums) |
| `batched_solve.rs` | `cusolver_batched_potrf_f64`, `cusolver_batched_potrs_f64` | Batched Cholesky solve (6×6) |
| `async_stream.rs` | `cuda_stream_*`, `cuda_event_*`, `cuda_memcpy_async_*` | Async transfers, pinned memory |

---

## Key Constants

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `MAX_NEIGHBORS` | 8 | `gpu_pipeline_kernels.rs:15`, `derivatives/gpu.rs:27` | Max voxels per radius search |
| `DERIVATIVE_OUTPUT_SIZE` | 28 | `derivatives/gpu.rs:32` | Per-point output: 1 score + 6 gradient + 21 Hessian |
| `DEFAULT_NUM_CANDIDATES` | 8 | `gpu_pipeline_kernels.rs:18` | Line search candidates |
| `BLOCK_SIZE` | 256 | `graph_ndt.rs:64`, `batch_persistent_ndt.cu:29` | Threads per block for compute kernels |
| `GRAPH_MAX_NEIGHBORS` | 8 | `ndt_graph_common.cuh` | Hash-table neighbor limit |

---

## Execution Pipelines

### Newton Iteration (Phase 24 Graph Pipeline)

```
K1 (init) → [ K2 (compute) → K3 (solve) → K4 (linesearch) → K5 (update) ] × max_iter
                                              ↑ optional
```

Captured as a CUDA graph after first execution; subsequent iterations replay the graph with near-zero launch overhead.

### Voxel Grid Construction (Zero-Copy Pipeline)

```
compute_morton_codes → pack_morton → CUB radix sort → detect_boundaries
→ CUB inclusive_sum → CUB select_flagged → segmented_reduce (means)
→ segmented_reduce (cov_sums) → finalize_voxels_cpu (eigendecomp)
→ voxel_hash_build
```

### Batched Scoring (Initial Pose)

```
compute_transforms_batch → transform_points_batch → compute_scores_batch
→ CUB segmented_reduce → per-pose totals
```
