# Phase 25: Code Restructure & Quality

**Status**: Planned
**Date**: 2026-01-28 (updated 2026-02-25)

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

## Current Structure

```
src/cuda_ndt_matcher/src/
├── main.rs              (1,934 lines) ← TOO BIG
├── covariance.rs        (703 lines)
├── diagnostics.rs       (446 lines)
├── dual_ndt_manager.rs  (468 lines)
├── initial_pose.rs      (527 lines)
├── map_module.rs        (836 lines)   ← Contains two distinct components
├── ndt_manager.rs       (554 lines)
├── nvtl.rs              (413 lines)
├── params.rs            (440 lines)
├── particle.rs          (79 lines)    ← Only used by initial_pose
├── pointcloud.rs        (419 lines)   ← Has both CPU and GPU paths
├── pose_buffer.rs       (464 lines)
├── scan_queue.rs        (457 lines)
├── tf_handler.rs        (396 lines)
├── tpe.rs               (303 lines)   ← Only used by initial_pose
└── visualization.rs     (709 lines)

Total: 9,148 lines across 16 files
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

### 25.4 Split main.rs

Split the 1,934-line `main.rs` into the `node/` module hierarchy.

**Criteria**:
- [ ] **node/state.rs**: `NdtScanMatcherNode` struct definition extracted (lines 129-167)
- [ ] **node/publishers.rs**: `DebugPublishers` struct, `publish_tf()`, marker helpers extracted (lines 84-123, 1507-1678)
- [ ] **node/services.rs**: `on_ndt_align()`, `on_map_update()`, `on_map_received()` extracted (lines 1684-1903)
- [ ] **node/processing.rs**: core alignment logic extracted from `on_points()` — pose interpolation, alignment execution, convergence gating, covariance estimation
- [ ] **node/callbacks.rs**: remaining `on_points()` structure — point cloud conversion, sensor frame transform, calls to processing, debug publishing
- [ ] **node/init.rs**: `NdtScanMatcherNode::new()` extracted (lines 170-617) — parameter loading, publisher/subscription/service creation
- [ ] **main.rs** reduced to ~50 lines (entry point only)
- [ ] All tests pass
- [ ] No functionality changes

---

### 25.5 Reorganize Module Structure

Move flat files into hierarchical directories using `git mv`.

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
| `pointcloud.rs`       | `io/pointcloud/`                 |
| `params.rs`           | `io/params.rs`                   |
| `diagnostics.rs`      | `io/diagnostics.rs`              |
| `visualization.rs`    | `visualization/markers.rs`       |

**Criteria**:
- [ ] All files moved to new locations with `git mv`
- [ ] `mod` declarations and `use` imports updated throughout
- [ ] Module visibility set: `node/` internal, others `pub(crate)` or `pub` as appropriate
- [ ] `map_module.rs` split into `map/tiles.rs` (`MapUpdateModule`) and `map/loader.rs` (`DynamicMapLoader`)
- [ ] All tests pass
- [ ] No functionality changes

---

### 25.6 Split Dual CPU/GPU Implementations

Make CPU and GPU code paths explicit in modules that have both.

**Criteria**:
- [ ] **pointcloud split**: `io/pointcloud/cpu.rs` (conversion, filtering) and `io/pointcloud/gpu.rs` (GPU-accelerated filtering) created; `io/pointcloud/mod.rs` re-exports and auto-selects
- [ ] CPU/GPU paths clearly identifiable by file location
- [ ] All tests pass

---

### 25.7 Error Handling Audit

Improve error handling in the `ndt_cuda` library crate (307 `unwrap()` calls).

**Criteria**:
- [ ] `unwrap()` calls in `ndt_cuda/src/**/*.rs` audited
- [ ] Public API paths use `?` + `context()` instead of `unwrap()`
- [ ] `unwrap()` retained only where infallibility is proven by construction (with a comment explaining why)
- [ ] Test code may retain `unwrap()` — production code should not
- [ ] All tests pass

---

### 25.8 Test & Build Hygiene

Improve test visibility, feature flag ergonomics, type safety, and clean up tech debt.

**Criteria**:
- [ ] **Test skips**: `require_cuda!()` macro replaced or augmented with `#[ignore = "requires CUDA"]` where appropriate; CI output distinguishes skipped from passed
- [ ] **Feature flags**: nested `#[cfg]` blocks in `on_points()` alignment path simplified (runtime `if cfg!()` or trait dispatch); feature dependency chain in `cuda_ndt_matcher/Cargo.toml` documented with comments; all feature combinations build (`default`, `profiling`, `debug`)
- [ ] **`as` casts at boundaries**: casts on external input (PointCloud2 parsing, ROS params) replaced with `try_into().context()?` or explicit bounds checks; internal casts left as-is
- [ ] **Stale TODOs**: each TODO evaluated — resolve, convert to GitHub issue, or prefix with `TECH-DEBT:`; no bare `TODO` in production code paths (test-only TODOs acceptable)
- [ ] All tests pass

## Module Classification

### GPU Path (alignment/)

| Module | Purpose |
|--------|---------|
| `alignment/manager.rs` | NDT alignment via ndt_cuda |
| `alignment/dual_manager.rs` | Non-blocking dual NDT |
| `alignment/covariance.rs` | Covariance estimation (orchestrates GPU batch) |
| `alignment/batch.rs` | Scan queue for batch GPU processing |

### CPU Modules

| Module | Purpose |
|--------|---------|
| `map/` | Tile management, map loading |
| `transform/` | TF buffer, pose interpolation, conversion utils |
| `scoring/` | NVTL reference implementation |
| `io/params.rs` | Parameter loading |
| `io/diagnostics.rs` | ROS diagnostics |
| `io/debug_writer.rs` | Centralized debug JSONL output |
| `visualization/` | Debug markers and clouds |

### Mixed CPU/GPU

| Module | Purpose |
|--------|---------|
| `initial_pose/` | CPU orchestrator for GPU batch alignment |
| `io/pointcloud/` | Explicit CPU/GPU filtering |
| `node/` | ROS orchestration |

## Implementation Order

| Sub-phase | Depends On | Effort |
|-----------|------------|--------|
| 25.1 Metadata & Cargo hygiene | — | 30 min |
| 25.2 Code deduplication | — | 6 hours |
| 25.3 Lint & visibility | — | 3 hours |
| 25.4 Split main.rs | 25.2 | 4 hours |
| 25.5 Reorganize modules | 25.4 | 3 hours |
| 25.6 Split CPU/GPU | 25.5 | 2 hours |
| 25.7 Error handling audit | — | 6 hours |
| 25.8 Test & build hygiene | 25.4 | 4 hours |

**Total estimated effort**: ~28 hours

Sub-phases 25.1–25.3 and 25.7 have no dependencies and can be done in parallel or any order. Sub-phases 25.4–25.6 are sequential (each builds on the previous). 25.8 depends on 25.4 because feature-flag cleanup touches the restructured callbacks.

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

- [ ] `main.rs` reduced to ~50 lines
- [ ] All modules in logical hierarchical structure
- [ ] CPU/GPU paths clearly identifiable
- [ ] `just build` succeeds
- [ ] `just test` passes (all 417+ tests)
- [ ] `just lint` passes with no new warnings
- [ ] No `#![allow(dead_code)]` at module level
- [ ] No `#[allow(clippy::too_many_arguments)]` on `on_points()`
- [ ] Zero bare `TODO` in production code paths
- [ ] `package.xml` files have real maintainer info
- [ ] No functionality changes
