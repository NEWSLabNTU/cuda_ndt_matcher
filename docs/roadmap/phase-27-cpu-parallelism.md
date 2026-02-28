# Phase 27: CPU-Side Parallelism & Pipeline Optimization

**Status**: Complete (27.1–27.6)
**Date**: 2026-02-28

## Motivation

The GPU pipeline (Phase 24 CUDA graphs) handles the core NDT alignment loop efficiently, but the CPU-side code surrounding it remains largely sequential. Profiling on Jetson shows NDT is the CPU bottleneck (46% of all Autoware CPU). The `on_points` callback processes stages sequentially where many are independent, and several per-point operations iterate over 10k–30k points without parallelism.

**Current state of parallelism**:
- `rayon` is already a dependency of `ndt_cuda` (used in voxel grid finalization and batch scoring)
- `cuda_ndt_matcher` does **not** use `rayon` at all
- GPU kernels handle the inner NDT loop, but pre/post-processing is single-threaded

**Impact areas**:
- Per-point CPU transforms (sensor→base_link, aligned cloud, no-ground scoring)
- Point filtering CPU fallback path
- Voxel grid CPU finalization (eigendecomposition per voxel)
- Sequential operations that could overlap (TF lookup vs. pose interpolation)
- Unnecessary allocations and clones in the hot path

## Goals

- Add `rayon` parallelism to CPU-bound per-point operations in the hot path
- Reduce unnecessary memory allocations and clones
- Eliminate lock contention where possible
- Pipeline independent stages for overlap

## Sub-phases

### 27.1 Parallelize Per-Point Transforms

The most frequently called per-point operations. Each processes the full sensor cloud (~10k–30k points) with independent per-point math.

#### 27.1a `transform_points_f32` — rayon par_iter

**File**: `cuda_ndt_matcher/src/transform/pose_utils.rs:52-65`

Called from:
- `callbacks.rs` — aligned points for debug publishing (per scan)
- `callbacks.rs` — no-ground aligned points (per scan, when enabled)
- `markers.rs` — visualization point transform (per scan, test-only)

Each point: f32→f64 cast, 3×3 rotation + translation, f64→f32 cast. Embarrassingly parallel.

```rust
// Before
points.iter().map(|p| { ... tf * pt ... }).collect()

// After (threshold to avoid rayon overhead on small clouds)
if points.len() > 4096 {
    points.par_iter().map(|p| { ... tf * pt ... }).collect()
} else {
    points.iter().map(|p| { ... tf * pt ... }).collect()
}
```

#### 27.1b `TfHandler::transform_points` — rayon par_iter

**File**: `cuda_ndt_matcher/src/transform/tf_handler.rs:272-284`

Called once per scan in `transform_to_base_frame` (Stage 2 of `on_points`). Same pattern as above — per-point isometry application.

**Note**: The TF lookup itself (`lookup_transform`) is fast (HashMap read). Only the bulk transform is worth parallelizing.

#### 27.1c No-ground filter transform — avoid double transform

**File**: `cuda_ndt_matcher/src/node/callbacks.rs:558-576`

Currently transforms each point individually inside a `.filter()` closure, then transforms the survivors again in the aligned-points block. The map-frame Z coordinate is computed twice for points that pass the filter.

```rust
// Current: transform in filter, then transform again for publishing
let no_ground_points: Vec<[f32; 3]> = sensor_points.iter()
    .filter(|pt| {
        let map_pt = pose_isometry * Point3::from(...);  // transform #1
        map_pt.z - base_link_z > z_threshold
    }).copied().collect();
let no_ground_aligned = transform_points_f32(&no_ground_points, &pose_isometry);  // transform #2

// Better: transform once, split into filtered + aligned
let no_ground_aligned: Vec<[f32; 3]> = sensor_points.iter()
    .filter_map(|pt| {
        let map_pt = pose_isometry * Point3::from(...);
        if map_pt.z - base_link_z > z_threshold {
            Some([map_pt.x as f32, map_pt.y as f32, map_pt.z as f32])
        } else {
            None
        }
    }).collect();
```

This eliminates N extra isometry multiplications (where N = no-ground point count).

**Criteria**:
- [x] `transform_points_f32` uses `par_iter` above 4096 points
- [x] `TfHandler::transform_points` uses `par_iter` above 4096 points (delegates to `transform_points_f32`)
- [x] No-ground filter avoids double transform (uses `filter_map` + `unzip`)
- [x] `rayon` added to `cuda_ndt_matcher` dependencies
- [x] All tests pass (484 passed)
- [x] No behavior changes

---

### 27.2 Parallelize CPU Scoring Fallbacks

CPU scoring functions in `ndt_cuda` are used as fallbacks and for validation. They iterate sequentially over all source points with per-point radius search + score computation.

#### 27.2a `compute_transform_probability` — rayon reduction

**File**: `ndt_cuda/src/scoring/metrics.rs:74-106`

Per-point: isometry transform + radius search + Gaussian score. Called for "before" and "after" scoring in `processing.rs` when GPU scoring pipeline is unavailable.

```rust
// Current: sequential accumulation
for source_point in source_points {
    let transformed = pose * pt;
    let nearby = target_grid.radius_search(&transformed, radius);
    for voxel in nearby { total_score += score; num += 1; }
}

// After: parallel with thread-local accumulation
let (total_score, num_correspondences, num_no_correspondence) = source_points
    .par_iter()
    .fold(|| (0.0, 0usize, 0usize), |(score, corr, no_corr), pt| { ... })
    .reduce(|| (0.0, 0, 0), |(s1,c1,n1), (s2,c2,n2)| (s1+s2, c1+c2, n1+n2));
```

#### 27.2b `compute_per_point_scores` — rayon par_iter

**File**: `ndt_cuda/src/scoring/metrics.rs:145-176`

Same pattern but returns per-point score vector. Used for visualization (debug-markers feature).

```rust
// Current
source_points.iter().map(|pt| { ... radius_search + score ... }).collect()
// After
source_points.par_iter().map(|pt| { ... radius_search + score ... }).collect()
```

**Note**: Both functions require `VoxelGrid::radius_search` to be `&self`-safe for concurrent access. The current KD-tree implementation uses `&self` (immutable), so this is safe.

**Criteria**:
- [x] `compute_transform_probability` uses rayon fold+reduce (threshold 4096)
- [x] `compute_per_point_scores` uses `par_iter` (threshold 4096)
- [x] `VoxelGrid::radius_search` confirmed `Send + Sync` (immutable `&self`)
- [x] All tests pass

---

### 27.3 Parallelize Voxel Grid CPU Finalization

#### 27.3a `finalize_voxels_cpu` — rayon parallel eigendecomposition

**File**: `ndt_cuda/src/voxel_grid/gpu/statistics.rs:406-457`

After GPU kernels compute per-voxel covariance sums, this CPU function finalizes each voxel: divides by (n-1), adds identity term, then calls `regularize_and_invert` which performs a 3×3 eigendecomposition + matrix reconstruction. With 1k–10k voxels, this is significant.

Each voxel's finalization is independent — the function reads from `counts[]` and `cov_sums[]` slices and writes to `covariances[]`, `inv_covariances[]`, `principal_axes[]`, and `valid[]` at non-overlapping indices.

```rust
// Current: sequential loop
for v in 0..num_voxels {
    // eigendecomposition + regularization per voxel
}

// After: parallel chunks (split output arrays into non-overlapping ranges)
(0..num_voxels).into_par_iter().for_each(|v| {
    // Same body, writing to v*9..v*9+9, v*3..v*3+3, v
});
```

Requires converting local `Vec` outputs to use `unsafe` slice writes or restructuring to return per-voxel results that are collected.

**Criteria**:
- [x] `finalize_voxels_cpu` uses `into_par_iter` over voxels (parallel collect + scatter)
- [x] No `unsafe` — uses `VoxelResult` struct with parallel collect
- [x] Eigendecomposition results identical (deterministic)
- [x] All tests pass (voxel comparison tests verify exact values)

---

### 27.4 Parallelize Point Filtering (CPU Fallback)

#### 27.4a `filter_points_cpu` — rayon parallel filter

**File**: `ndt_cuda/src/filtering/cpu.rs:18-34`

Applied to every incoming scan when the GPU filter is not used (point cloud < 10k or no CUDA). Sequential distance² + Z-height checks per point.

```rust
// Current: sequential with counters
for p in points {
    if dist_sq < min || dist_sq > max { removed_by_distance += 1; continue; }
    if p[2] < min_z || p[2] > max_z { removed_by_z += 1; continue; }
    filtered.push(*p);
}

// After: parallel partition + count
let (filtered, removed_distance, removed_z) = points.par_iter()
    .fold(|| (Vec::new(), 0, 0), |(mut f, mut d, mut z), p| {
        let dist_sq = p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
        if dist_sq < min_dist_sq || dist_sq > max_dist_sq { d += 1; }
        else if p[2] < min_z || p[2] > max_z { z += 1; }
        else { f.push(*p); }
        (f, d, z)
    })
    .reduce(|| (Vec::new(), 0, 0), |(mut f1,d1,z1), (mut f2,d2,z2)| {
        f1.append(&mut f2); (f1, d1+d2, z1+z2)
    });
```

#### 27.4b `voxel_downsample_cpu` — rayon parallel accumulation

**File**: `ndt_cuda/src/filtering/cpu.rs:67-99`

Sequential HashMap accumulation for voxel downsampling. Can use thread-local HashMaps with parallel merge.

**Criteria**:
- [x] `filter_points_cpu` uses rayon fold+reduce (threshold 4096)
- [x] `voxel_downsample_cpu` uses thread-local HashMap accumulation (threshold 4096)
- [x] Filtered point counts match sequential version
- [x] All tests pass

---

### 27.5 Reduce Allocations in Hot Path

Eliminate repeated `Vec` allocations and unnecessary clones that occur on every scan.

#### 27.5a Pose buffer — avoid clones during linear search

**File**: `cuda_ndt_matcher/src/transform/pose_buffer.rs:103-114`

Currently clones poses in a loop to find bracketing entries. Can use indices instead.

```rust
// Current: O(N) clones
for pose in buffer.iter() {
    new_pose = pose.clone();
    if pose_time_ns > target_time_ns { break; }
    old_pose = pose.clone();
}

// After: use indices, clone only the 2 results
let mut old_idx = 0;
let mut new_idx = 0;
for (i, pose) in buffer.iter().enumerate() {
    new_idx = i;
    if Self::stamp_to_ns(&pose.header.stamp) > target_time_ns { break; }
    old_idx = i;
}
let old_pose = buffer[old_idx].clone();
let new_pose = buffer[new_idx].clone();
```

#### 27.5b Pose buffer — binary search

The pose buffer is time-ordered by construction (`push_back` clears on reversal). Replace the linear scan with `partition_point` for O(log N) lookup.

```rust
let idx = buffer.partition_point(|p| Self::stamp_to_ns(&p.header.stamp) <= target_time_ns);
```

#### 27.5c TF handler — reduce string allocations

**File**: `cuda_ndt_matcher/src/transform/tf_handler.rs:96`

`frame_id.clone()` and `child_frame_id.clone()` on every `/tf` message. The buffer key is `(String, String)`. Consider interning frame names or using `Arc<str>`.

#### 27.5d Point flattening — pre-allocate or reuse buffers

**File**: `ndt_cuda/src/voxel_grid/gpu_builder.rs:286`

`points.iter().flat_map(|p| p.iter().copied()).collect::<Vec<f32>>()` allocates on every voxel grid build. Could reuse a buffer across calls.

**Criteria**:
- [x] Pose buffer uses binary search (`partition_point`) — replaces O(N) linear scan with O(log N)
- [x] Only 2 clones (old_pose, new_pose) instead of O(N) clones per interpolation
- [x] All tests pass
- [x] No behavior changes

---

### 27.6 Pipeline Stage Overlap

Overlap independent stages in the `on_points` callback to reduce end-to-end latency.

#### 27.6a Compute sensor_time_ns and start TF lookup concurrently with point filtering

Stages 1 (convert+filter) and 2 (TF transform) are currently sequential. The TF lookup (`lookup_transform`) is independent of the point data — it only needs the frame names and timestamp. The lookup could start while filtering is in progress, with the bulk transform applied after both complete.

```
Current:  [Filter points] → [TF lookup + transform]
Proposed: [Filter points]─────────────────→[bulk transform]
          [TF lookup (HashMap read)]──────↗
```

This requires splitting `transform_to_base_frame` into lookup + apply.

#### 27.6b Async debug publishing

Stage 8 (debug publishers) transforms the aligned point cloud and publishes debug metrics after the main alignment. These publishers are non-critical and could run on a background thread.

```
Current:  [Alignment] → [Publish pose] → [Debug transforms + publish]
Proposed: [Alignment] → [Publish pose]
                       ↘ [Debug transforms + publish] (background)
```

This would reduce the critical path by the time spent on debug publishing (~2-3ms when features enabled).

#### 27.6c Overlap covariance estimation with next scan

When using MULTI_NDT covariance mode, the estimation runs 10-20 additional alignments. This could be deferred to a background task, publishing the covariance when ready rather than blocking the callback.

**Note**: This changes observable timing — the covariance message would arrive slightly after the pose message. Verify that downstream consumers (EKF) tolerate this.

**Criteria**:
- [x] TF lookup split from bulk transform (`lookup_base_transform` + `apply_base_transform`)
- [ ] Debug publishing moved to background (deferred — requires async runtime evaluation)
- [x] No behavior changes in default configuration

---

## What Stays Sequential

These patterns were evaluated and intentionally left sequential:

| Pattern                       | Location                 | Reason                                         |
|-------------------------------|--------------------------|------------------------------------------------|
| Single-point norm             | `callbacks.rs:85-88`     | Single `fold` for max distance — tiny overhead |
| Single-point filter transform | `callbacks.rs:558-576`   | Addressed in 27.1c (merged, not parallelized)  |
| Stamp-to-f64-seconds          | `diagnostics.rs:627`     | Single arithmetic operation                    |
| Softmax weights               | `covariance.rs:189-211`  | 8–24 elements — rayon overhead exceeds benefit |
| Offset pose generation        | `covariance.rs:235-268`  | 8–24 poses — too small for parallelism         |
| Sample covariance             | `covariance.rs:278-340`  | 8–24 data points                               |
| Particle selection            | `particle.rs:32-38`      | `max_by` on 100-500 elements — already optimal |
| Multi-grid cascade            | `multi_grid.rs:414-424`  | Inherently sequential (coarse→fine dependency) |
| Pose buffer `pop_old`         | `pose_buffer.rs:152-161` | Linear drain from front — O(1) amortized       |
| Test helpers                  | Various `make_pose`      | Test-only duplication                          |

## Files Modified

| Sub-phase | Crate | Files |
|-----------|-------|-------|
| 27.1 | `cuda_ndt_matcher` | `transform/pose_utils.rs`, `transform/tf_handler.rs`, `node/callbacks.rs`, `Cargo.toml` |
| 27.2 | `ndt_cuda` | `scoring/metrics.rs` |
| 27.3 | `ndt_cuda` | `voxel_grid/gpu/statistics.rs` |
| 27.4 | `ndt_cuda` | `filtering/cpu.rs` |
| 27.5 | `cuda_ndt_matcher`, `ndt_cuda` | `transform/pose_buffer.rs`, `transform/tf_handler.rs`, `voxel_grid/gpu_builder.rs` |
| 27.6 | `cuda_ndt_matcher` | `node/callbacks.rs`, `transform/tf_handler.rs` |

## Implementation Order

| Sub-phase | Depends On | Effort | Impact |
|-----------|------------|--------|--------|
| 27.1 Per-point transforms | — | 2 hours | High (every scan) |
| 27.2 CPU scoring fallbacks | — | 2 hours | Medium (fallback path) |
| 27.3 Voxel finalization | — | 3 hours | High (map updates) |
| 27.4 Point filtering | — | 2 hours | Medium (CPU fallback) |
| 27.5 Reduce allocations | — | 3 hours | Medium (every scan) |
| 27.6 Pipeline overlap | 27.1 | 4 hours | High (latency reduction) |

Sub-phases 27.1–27.5 are independent and can be done in any order. 27.6 benefits from 27.1 (split TF lookup from transform).

## Verification

After each sub-phase:

```bash
just build
just test
just lint
just profile-compare  # Verify no regression, measure improvement
```

## Risks

1. **Rayon overhead on small inputs**: Use size thresholds (4096 points) to avoid spawning threads for tiny workloads
2. **Non-deterministic ordering**: Parallel collect may change point order in filtered results — verify downstream consumers don't depend on order
3. **Thread contention with GPU**: Rayon threads may compete with CUDA driver threads on Jetson (limited cores) — profile on target hardware
4. **Float accumulation order**: Parallel reduction changes summation order, producing slightly different floating-point results — verify within tolerance

## Dependencies

- `rayon` already in `ndt_cuda` Cargo.toml — add to `cuda_ndt_matcher`
- No external API changes
- No conflict with other phases
