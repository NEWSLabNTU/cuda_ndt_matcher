# Phase 25: Code Restructure & Quality

**Status**: In Progress (25.1–25.9 complete, 25.10 pending)
**Date**: 2026-01-28 (updated 2026-02-28)

## Motivation

The codebase has grown organically and needs both structural reorganization and quality improvements:

1. **main.rs is too large** (1,934 lines) — difficult to navigate and maintain
2. **Flat module structure** — unclear relationships between modules (e.g., `tpe.rs` only used by `initial_pose.rs`)
3. **CPU/GPU paths not explicit** — hard to identify which code runs on GPU vs CPU
4. **Excessive `unwrap()` calls** (307 in `ndt_cuda`) — risk panics in production
5. **No `pub(crate)` scoping** — internal APIs are fully public
6. **Code duplication** — PointCloud2 construction, quaternion conversion, debug I/O repeated across files
7. **Module-level `#[allow(dead_code)]`** — masks real dead code
8. **Silent test skips** — `require_cuda!()` macro hides whether tests actually ran in CI

## Goals

- Fix low-hanging metadata and hygiene issues first
- Reduce duplication before moving files (smaller diffs)
- Split `main.rs` into manageable modules
- Create hierarchical module structure reflecting actual dependencies
- Make CPU/GPU implementations explicit where both exist
- Improve error handling and test visibility across all crates

## Current Structure (post-25.6)

```
src/cuda_ndt_matcher/src/
├── main.rs                    (33 lines)  - Entry point only
│
├── node/                      - ROS node components
│   ├── mod.rs
│   ├── state.rs               - NdtScanMatcherNode struct
│   ├── init.rs                - new() initialization
│   ├── callbacks.rs           - on_points() callback
│   ├── services.rs            - Service handlers
│   ├── publishers.rs          - Debug publishers, TF
│   └── processing.rs          - Alignment processing logic
│
├── alignment/                 - NDT alignment (GPU path)
│   ├── mod.rs
│   ├── manager.rs             ← ndt_manager.rs
│   ├── dual_manager.rs        ← dual_ndt_manager.rs
│   ├── covariance.rs          ← covariance.rs
│   └── batch.rs               ← scan_queue.rs
│
├── initial_pose/              - Initial pose estimation
│   ├── mod.rs
│   ├── estimator.rs           ← initial_pose.rs
│   ├── tpe.rs                 ← tpe.rs
│   └── particle.rs            ← particle.rs
│
├── map/                       - Map management (CPU)
│   ├── mod.rs
│   ├── tiles.rs               ← map_module.rs (MapUpdateModule)
│   └── loader.rs              ← map_module.rs (DynamicMapLoader)
│
├── transform/                 - Spatial transforms (CPU)
│   ├── mod.rs
│   ├── tf_handler.rs          ← tf_handler.rs
│   ├── pose_buffer.rs         ← pose_buffer.rs
│   └── pose_utils.rs          ← pose_utils.rs
│
├── scoring/                   - Scoring reference (CPU)
│   ├── mod.rs
│   └── nvtl.rs                ← nvtl.rs
│
├── io/                        - I/O utilities
│   ├── mod.rs
│   ├── pointcloud/            - Explicit CPU/GPU split
│   │   ├── mod.rs             re-exports from cpu + gpu
│   │   ├── cpu.rs             PointCloud2 parsing/construction
│   │   └── gpu.rs             GPU-accelerated filtering + CPU fallback
│   ├── params.rs              ← params.rs
│   ├── diagnostics.rs         ← diagnostics.rs
│   └── debug_writer.rs        ← debug_writer.rs
│
└── visualization/             - Debug visualization (CPU)
    ├── mod.rs
    └── markers.rs             ← visualization.rs

33 files, 8 module declarations in main.rs
```

## Target Structure

```
src/cuda_ndt_matcher/src/
├── main.rs                    (~50 lines)   - Entry point only
├── lib.rs                     (~30 lines)   - Crate re-exports
│
├── node/                      - ROS node components
│   ├── mod.rs                 (~30 lines)
│   ├── state.rs               (~150 lines) - NdtScanMatcherNode struct
│   ├── init.rs                (~450 lines) - new() initialization
│   ├── callbacks.rs           (~550 lines) - on_points() callback
│   ├── services.rs            (~250 lines) - Service handlers
│   ├── publishers.rs          (~200 lines) - Debug publishers, TF
│   └── processing.rs          (~300 lines) - Alignment processing logic
│
├── alignment/                 - NDT alignment (GPU path)
│   ├── mod.rs
│   ├── manager.rs             (554 lines)  ← ndt_manager.rs
│   ├── dual_manager.rs        (468 lines)  ← dual_ndt_manager.rs
│   ├── covariance.rs          (703 lines)  ← covariance.rs
│   └── batch.rs               (457 lines)  ← scan_queue.rs
│
├── initial_pose/              - Initial pose estimation
│   ├── mod.rs
│   ├── estimator.rs           (527 lines)  ← initial_pose.rs
│   ├── tpe.rs                 (303 lines)  ← tpe.rs (PUBLIC)
│   └── particle.rs            (79 lines)   ← particle.rs
│
├── map/                       - Map management (CPU)
│   ├── mod.rs
│   ├── tiles.rs               (~500 lines) ← map_module.rs (MapUpdateModule)
│   └── loader.rs              (~340 lines) ← map_module.rs (DynamicMapLoader)
│
├── transform/                 - Spatial transforms (CPU)
│   ├── mod.rs
│   ├── tf_handler.rs          (396 lines)  ← tf_handler.rs
│   ├── pose_buffer.rs         (464 lines)  ← pose_buffer.rs
│   └── pose_utils.rs          (~60 lines)  ← NEW: shared conversion helpers
│
├── scoring/                   - Scoring reference (CPU)
│   ├── mod.rs
│   └── nvtl.rs                (413 lines)  ← nvtl.rs
│
├── io/                        - I/O utilities
│   ├── mod.rs
│   ├── pointcloud/            - Explicit CPU/GPU split
│   │   ├── mod.rs
│   │   ├── cpu.rs             ← CPU conversion/filtering
│   │   └── gpu.rs             ← GPU-accelerated filtering
│   ├── params.rs              (440 lines)  ← params.rs
│   ├── diagnostics.rs         (446 lines)  ← diagnostics.rs
│   └── debug_writer.rs        (~50 lines)  ← NEW: centralized debug JSONL I/O
│
└── visualization/             - Debug visualization (CPU)
    ├── mod.rs
    └── markers.rs             (709 lines)  ← visualization.rs
```

## Sub-phases

### 25.1 Fix Metadata & Cargo Hygiene

Quick fixes that ship in package metadata or affect build consistency.

**Criteria**:
- [x] `src/cuda_ndt_matcher/package.xml` has real maintainer name and email (not `TODO`)
- [x] `src/cuda_ndt_matcher_launch/package.xml` has real maintainer name and email (not `TODO`)
- [x] Workspace dependency syntax normalized to one style across all `Cargo.toml` files
- [x] Doc examples in `ndt_cuda/src/lib.rs` changed from ` ```ignore ` to ` ```no_run ` where valid Rust (keeps ` ```ignore ` with comment where ROS context is needed)
- [x] `cargo doc` succeeds
- [x] `just build` succeeds

---

### 25.2 Code Deduplication & Extraction

Reduce duplication before the structural moves (smaller diffs in later sub-phases).

**Criteria**:
- [x] **Callback context struct**: `OnPointsContext` created holding all shared state; `on_points()` signature reduced to `(msg, ctx)`; `#[allow(clippy::too_many_arguments)]` removed
- [x] **Pose utilities**: `pose_utils.rs` created with `isometry_from_pose()`, `pose_from_isometry()`, `unit_quat_from_msg()`, `euler_from_pose()`; all duplicate conversion sites in `main.rs`, `covariance.rs`, `initial_pose.rs` updated
- [x] **Debug I/O**: `debug_writer.rs` created with `clear_debug_file()`, `append_debug_line()`, `write_init_to_tracking()`; all duplicate sites in `main.rs` and `initial_pose.rs` use it; `NDT_DEBUG_FILE` env var handling centralized
- [x] **PointCloud2 builder**: `xyz_fields()` and `encode_xyz_data()` helpers extracted; `to_pointcloud2()`, `to_pointcloud2_with_rgb()`, and test helper delegate to shared logic
- [x] All tests pass (`just test`) — 421 tests (355 ndt_cuda + 66 cuda_ffi)
- [x] No functionality changes

---

### 25.3 Lint & Visibility Hygiene

Tighten up compiler warnings and API surface.

**Criteria**:
- [x] `#![allow(dead_code)]` removed from all 8 modules (`covariance.rs`, `scan_queue.rs`, `ndt_manager.rs`, `diagnostics.rs`, `particle.rs`, `map_module.rs`, `params.rs`, `nvtl.rs`); individual `#[allow(dead_code)]` added only to items genuinely used via `Arc<Mutex<T>>` or closure captures; truly dead code removed (`ExecutionTimer`, `MapStats`, `get_stats`, `get_map_points_ref`, `params`, `estimate_covariance`); feature-gated items given proper `#[cfg]`; `nvtl.rs` retains module-level allow as CPU-only reference implementation
- [x] All `pub fn` / `pub struct` in `cuda_ndt_matcher/src/*.rs` audited; internal-only items changed to `pub(crate)` across all 17 module files
- [x] `just lint` passes with no new warnings (3 pre-existing clippy warnings in visualization.rs unchanged)

---

### 25.4 Split main.rs ✓

Split the 1,934-line `main.rs` into the `node/` module hierarchy.

**Completed**: 2026-02-27

**Criteria**:
- [x] **node/state.rs** (145 lines): `NdtScanMatcherNode`, `OnPointsContext`, `DebugPublishers` structs, type aliases, `NODE_NAME` constant
- [x] **node/publishers.rs** (185 lines): `publish_tf()`, `create_pose_marker()`, `create_pose_history_markers()` as free functions
- [x] **node/services.rs** (258 lines): `on_ndt_align()`, `on_map_received()`, `on_map_update()` as free functions; `set_map()` as impl method
- [x] **node/processing.rs** (227 lines): `AlignmentOutput` struct, `run_alignment()` — alignment execution, score computation, convergence gating
- [x] **node/callbacks.rs** (705 lines): `on_points()` — point cloud conversion, sensor frame transform, calls to processing, covariance estimation, pose/TF publishing, debug publishing
- [x] **node/init.rs** (482 lines): `NdtScanMatcherNode::new()` — parameter loading, publisher/subscription/service creation
- [x] **main.rs** reduced to 42 lines (entry point only)
- [x] All tests pass (419 tests: 355 ndt_cuda + 64 cuda_ndt_matcher)
- [x] No functionality changes
- [x] Bonus: fixed pre-existing `#[cfg(feature = "debug-iteration")]` → `"debug-iterations"` typo

---

### 25.5 Reorganize Module Structure ✓

Move flat files into hierarchical directories using `git mv`.

**Completed**: 2026-02-27

| Current               | New Location                     |
|-----------------------|----------------------------------|
| `ndt_manager.rs`      | `alignment/manager.rs`           |
| `dual_ndt_manager.rs` | `alignment/dual_manager.rs`      |
| `covariance.rs`       | `alignment/covariance.rs`        |
| `scan_queue.rs`       | `alignment/batch.rs`             |
| `initial_pose.rs`     | `initial_pose/estimator.rs`      |
| `tpe.rs`              | `initial_pose/tpe.rs`            |
| `particle.rs`         | `initial_pose/particle.rs`       |
| `map_module.rs`       | `map/tiles.rs` + `map/loader.rs` |
| `tf_handler.rs`       | `transform/tf_handler.rs`        |
| `pose_buffer.rs`      | `transform/pose_buffer.rs`       |
| `nvtl.rs`             | `scoring/nvtl.rs`                |
| `pointcloud.rs`       | `io/pointcloud.rs`               |
| `params.rs`           | `io/params.rs`                   |
| `diagnostics.rs`      | `io/diagnostics.rs`              |
| `visualization.rs`    | `visualization/markers.rs`       |

**Criteria**:
- [x] All 16 files moved to new locations with `git mv` (preserving history)
- [x] `mod` declarations and `use` imports updated throughout (~15 files)
- [x] Module visibility set: `node/` internal, others `pub(crate)`; 7 `mod.rs` files with re-exports
- [x] `map_module.rs` split into `map/tiles.rs` (`MapUpdateModule`, tests) and `map/loader.rs` (`DynamicMapLoader`)
- [x] `main.rs` reduced from 18 mod declarations to 8
- [x] All 64 tests pass, zero warnings
- [x] No functionality changes

---

### 25.6 Split Dual CPU/GPU Implementations ✓

Make CPU and GPU code paths explicit in modules that have both.

**Completed**: 2026-02-27

**Criteria**:
- [x] **pointcloud split**: `io/pointcloud/cpu.rs` (PointCloud2 parsing/construction, 2 tests) and `io/pointcloud/gpu.rs` (GPU-accelerated filtering with CPU fallback, 4 tests) created; `io/pointcloud/mod.rs` re-exports all public items
- [x] CPU/GPU paths clearly identifiable by file location
- [x] All 64 tests pass, zero warnings
- [x] No functionality changes — callers still use `pointcloud::from_pointcloud2()` etc. via re-exports

---

### 25.7 Error Handling Audit ✓

Improve error handling in the `ndt_cuda` library crate (307 `unwrap()` calls).

**Completed**: 2026-02-27

**Criteria**:
- [x] `unwrap()` calls in `ndt_cuda/src/**/*.rs` audited
- [x] Public API paths use `?` + `context()` instead of `unwrap()`
- [x] `unwrap()` retained only where infallibility is proven by construction (with a comment explaining why)
- [x] Test code may retain `unwrap()` — production code should not
- [x] All tests pass

---

### 25.8 Test & Build Hygiene ✓

Improve test visibility, feature flag ergonomics, type safety, and clean up tech debt.

**Completed**: 2026-02-27

**Criteria**:
- [x] **Test skips**: `require_cuda!()` macro changed to `panic!` so tests fail explicitly on non-CUDA systems instead of silently passing
- [x] **Feature flags**: nested `#[cfg]` blocks in `on_points()` alignment path simplified; `debug-iterations` implies `debug-output` so single feature gate suffices; all feature combinations build (`default`, `profiling`, `debug`)
- [x] **`as` casts at boundaries**: 11 `as i32` casts on ROS params replaced with `try_into().context()?`; u32 overflow in PointCloud2 parsing fixed; output casts documented with safety comments
- [x] **Stale TODOs**: production TODO in `main.rs` removed; `gpu_newton.rs` TODO prefixed with `TECH-DEBT:`; test-only TODO in `ndt.rs` kept
- [x] All tests pass

---

### 25.9 Remove Dead GPU Code ✓

Remove CUDA kernels and FFI bindings that were superseded by the graph-based pipeline (Phase 24) but never deleted.

**Completed**: 2026-02-28

**Dead code removed** (~2,241 LOC):

| File                             | LOC   | Why dead                                                          |
|----------------------------------|-------|-------------------------------------------------------------------|
| `csrc/persistent_ndt.cu`        | 1,209 | Cooperative groups approach; fails on Jetson, superseded by graph  |
| `csrc/texture_voxels.cu`        | 240   | Texture memory objects; never integrated into any pipeline         |
| `cuda_ffi/src/persistent_ndt.rs` | 455  | FFI bindings for `persistent_ndt.cu`                              |
| `cuda_ffi/src/texture.rs`       | 337   | FFI bindings for `texture_voxels.cu`                              |

**Verified alive** (NOT removed):

| Item | Why alive |
|------|-----------|
| `batch_persistent_ndt.cu` + `.rs` | Used by `batch_pipeline.rs` and `async_pipeline.rs` |
| `persistent_ndt_device.cuh` | Included by `batch_persistent_ndt.cu` and `ndt_graph_kernels.cu` |
| `jacobi_svd_6x6.cuh` | Included by `ndt_graph_kernels.cu` |
| All other `.cuh` headers | Included by live `.cu` files |
| Point-to-plane CubeCL kernels | Used in `runtime.rs` for `DistanceMetric::PointToPlane` |

**Criteria**:
- [x] 2 `.cu` files removed from `cuda_ffi/csrc/` and `build.rs`
- [x] 2 `.rs` FFI modules removed from `cuda_ffi/src/`, `lib.rs` updated
- [x] All re-exports in `cuda_ffi/src/lib.rs` updated (no dangling `pub use`)
- [x] `ndt_cuda` compiles — no references to removed items
- [x] All 484 tests pass (1 skipped: opt-in benchmark)
- [x] 2,241 LOC removed

---

### 25.10 Extract Stages in Large Functions

Break large functions into smaller private helpers by extracting logical stages. No behavior changes — purely readability refactoring.

#### `callbacks.rs::on_points()` (682 LOC → ~80 LOC orchestrator)

Extract 6 helpers within the `impl NdtScanMatcherNode` block:

| Helper                            | Lines   | What it does                                               |
|-----------------------------------|---------|------------------------------------------------------------|
| `convert_and_filter_points()`     | 27–65   | Parse PointCloud2, apply sensor filters                    |
| `transform_to_base_frame()`       | 67–105  | TF lookup from sensor_frame → base_link                    |
| `interpolate_initial_pose()`      | 117–167 | SmartPoseBuffer interpolation + validation                 |
| `update_map_if_needed()`          | 174–230 | Check map freshness, request tiles, apply pending updates  |
| `publish_converged_pose()`        | 303–368 | Covariance estimation + PoseStamped + TF + MULTI_NDT poses |
| `publish_debug_and_diagnostics()` | 370–707 | All debug metric publishers + diagnostics collection       |

The remaining `on_points()` becomes a ~80-line orchestrator calling these in sequence with early returns.

#### `init.rs::NdtScanMatcherNode::new()` (455 LOC → ~90 LOC orchestrator)

Extract 5 helpers:

| Helper                      | Lines   | What it does                                               |
|-----------------------------|---------|------------------------------------------------------------|
| `create_publishers()`       | 91–141  | Create QoS + all 20 publishers + DebugPublishers struct    |
| `create_batch_queue()`      | 143–236 | ScanQueue with alignment closure and result callback       |
| `create_subscriptions()`    | 238–351 | 4 subscriptions: points_raw, EKF pose, regularization, map |
| `create_services()`         | 353–449 | 3 services: trigger, ndt_align, map_update                 |
| `initialize_debug_output()` | 451–460 | Clear debug file (feature-gated)                           |

#### `solver.rs::align()` (169 LOC) and `align_with_debug()` (231 LOC)

These two functions duplicate 90% of the Newton loop logic. Extract shared helpers:

| Helper                            | What it does                                                                                                       |
|-----------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `build_result()`                  | Build `NdtResult` from current state (pose, score, hessian, oscillation) — currently repeated 3 times in `align()` |
| `compute_newton_step_and_check()` | Newton step + convergence/singular checks — shared between `align()` and `align_with_debug()`                      |

#### `pipeline.rs::build()` (235 LOC → ~60 LOC orchestrator)

Extract 3 helpers within `GpuVoxelGridBuilder`:

| Helper                        | Lines   | What it does                                                   |
|-------------------------------|---------|----------------------------------------------------------------|
| `prepare_centered_points()`   | 374–412 | Center points around grid centroid, build segment_starts array |
| `launch_statistics_kernels()` | 414–453 | Launch 3 CubeCL kernels (sums, means, covariances)             |
| `download_and_finalize()`     | 455–575 | Download GPU buffers, un-center means, CPU finalization        |

**Criteria**:
- [ ] `on_points()` body reduced to <100 LOC (sequential helper calls + early returns)
- [ ] `NdtScanMatcherNode::new()` body reduced to <100 LOC
- [ ] `align()` and `align_with_debug()` share `build_result()` helper (no code duplication)
- [ ] `pipeline.rs::build()` body reduced to <80 LOC
- [ ] All extracted functions are `fn` (private), not `pub`
- [ ] No behavior changes — identical output for identical input
- [ ] All tests pass
- [ ] `just lint` passes

## Module Classification

### GPU Path (alignment/)

| Module | Purpose |
|--------|---------|
| `alignment/manager.rs` | NDT alignment via ndt_cuda |
| `alignment/dual_manager.rs` | Non-blocking dual NDT |
| `alignment/covariance.rs` | Covariance estimation (orchestrates GPU batch) |
| `alignment/batch.rs` | Scan queue for batch GPU processing |

### CPU Modules

| Module               | Purpose                                         |
|----------------------|-------------------------------------------------|
| `map/`               | Tile management, map loading                    |
| `transform/`         | TF buffer, pose interpolation, conversion utils |
| `scoring/`           | NVTL reference implementation                   |
| `io/params.rs`       | Parameter loading                               |
| `io/diagnostics.rs`  | ROS diagnostics                                 |
| `io/debug_writer.rs` | Centralized debug JSONL output                  |
| `visualization/`     | Debug markers and clouds                        |

### Mixed CPU/GPU

| Module | Purpose |
|--------|---------|
| `initial_pose/` | CPU orchestrator for GPU batch alignment |
| `io/pointcloud/` | Explicit CPU/GPU filtering |
| `node/` | ROS orchestration |

## Implementation Order

| Sub-phase                               | Depends On | Effort  |
|-----------------------------------------|------------|---------|
| 25.1 Metadata & Cargo hygiene           | —          | 30 min  |
| 25.2 Code deduplication                 | —          | 6 hours |
| 25.3 Lint & visibility                  | —          | 3 hours |
| 25.4 Split main.rs                      | 25.2       | 4 hours |
| 25.5 Reorganize modules                 | 25.4       | 3 hours |
| 25.6 Split CPU/GPU                      | 25.5       | 2 hours |
| 25.7 Error handling audit               | —          | 6 hours |
| 25.8 Test & build hygiene               | 25.4       | 4 hours |
| 25.9 Remove dead GPU code               | —          | 2 hours |
| 25.10 Extract stages in large functions | 25.4       | 4 hours |

Sub-phases 25.9 and 25.10 are independent of each other. 25.9 touches `cuda_ffi` only. 25.10 touches `cuda_ndt_matcher` and `ndt_cuda`.

## Migration Strategy

1. **Preserve git history**: Use `git mv` for file moves
2. **Incremental changes**: One sub-phase at a time, verify builds
3. **Update imports**: Fix all `use` statements after each move
4. **Run tests**: Ensure all tests pass after each sub-phase

## Verification

After each sub-phase:

```bash
just build
just test
just lint
```

## Risks

1. **Import breakage**: Many files import from main.rs indirectly
2. **Circular dependencies**: Must carefully order module declarations
3. **Feature flags**: Some code is behind `#[cfg(feature = "...")]`

## Dependencies

- No external dependencies
- Internal refactoring only
- No API changes to ndt_cuda crate
- No conflict with Phase 23 (GPU Utilization)

## Success Criteria

- [x] `main.rs` reduced to ~50 lines (33 lines)
- [x] All modules in logical hierarchical structure (8 directories, 33 files)
- [x] CPU/GPU paths clearly identifiable (`io/pointcloud/cpu.rs` vs `gpu.rs`)
- [x] `just build` succeeds
- [x] `just test` passes (all 417+ tests)
- [x] `just lint` passes with no new warnings
- [x] No `#![allow(dead_code)]` at module level
- [x] No `#[allow(clippy::too_many_arguments)]` on `on_points()`
- [x] Zero bare `TODO` in production code paths (25.8)
- [x] `package.xml` files have real maintainer info
- [x] No functionality changes
- [x] Dead GPU code removed (2,241 LOC: 2 `.cu`, 2 FFI `.rs`) (25.9)
- [ ] `on_points()` body <100 LOC; `new()` body <100 LOC (25.10)
- [ ] `align()` and `align_with_debug()` share `build_result()` — no duplication (25.10)
