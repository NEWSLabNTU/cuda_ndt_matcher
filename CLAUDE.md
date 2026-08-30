# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

CUDA/Rust re-implementation of Autoware's `ndt_scan_matcher` using CubeCL for GPU compute.

**Target**: Autoware 1.5.0

**Reference implementation**: `tests/comparison/autoware_core/localization/autoware_ndt_scan_matcher/`

**Documentation**:
- `docs/reference/` - Architecture, kernel inventory, feature list
- `docs/performance/autoware-comparison.md` - CUDA vs Autoware performance data
- `docs/roadmap/` - Implementation phases and status
- `docs/guides/rosbag-replay-guide.md` - Rosbag testing guide

## Build Commands

**Always use justfile** (never run colcon directly):

```bash
just build    # colcon build with --release
just clean    # rm -rf build install log target
just lint     # Format check + clippy (requires build first)
just test     # Run tests (requires build first)
```

**Running cargo directly** (for specific tests):
```bash
cargo test -p ndt_cuda --lib test_name
```

## Running

```bash
# Demo mode with logging
just run-cuda      # CUDA NDT
just run-builtin   # Autoware NDT (baseline)
```

See `docs/guides/rosbag-replay-guide.md` for custom rosbag testing.

## Project Structure

```
src/
├── ndt_cuda/           # Core NDT library (CubeCL GPU kernels)
├── cuda_ffi/           # CUDA FFI bindings (CUB primitives)
├── cuda_ndt_matcher/   # ROS 2 node
└── cuda_ndt_matcher_launch/  # Launch files and config
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `NDT_USE_GPU=0` | Force CPU mode (default: 1 for GPU) |
| `NDT_DEBUG=1` | Enable debug JSONL output |
| `NDT_DEBUG_VPP=1` | Log voxel-per-point distribution |
| `NDT_DEBUG_COV=1` | Compare GPU vs CPU covariance (output via tracing::debug) |
| `NDT_DUMP_VOXELS=1` | Dump voxel data to JSON for comparison |
| `NDT_DUMP_VOXELS_FILE` | Output path (default: `/tmp/ndt_cuda_voxels.json`) |
| `CUDA_ARCHS` | Architectures to build kernels for: `deploy` (default), `native`, `all`, or a list such as `87,120` |
| `CUDA_ARCH` | A single compute capability; overrides `CUDA_ARCHS` |
| `CUDA_PATH` / `CUDA_HOME` | Preferred CUDA toolkit; used whenever it can build the requested architectures |

**Pipeline config**: `PipelineV2Config::enable_debug = true` collects per-iteration debug data (score, gradient, Hessian, step size) from the graph kernels with zero overhead when disabled.

## CUDA Toolkits and Architectures

A kernel image only runs on the architecture it was built for, so `cuda_ffi`
builds a fat binary. The default set, `deploy`, covers every card this project
targets -- sm_86 (RTX 3090), sm_87 (AGX Orin), sm_89 (RTX 4090), sm_90 (H100)
and sm_120 (RTX 5090) -- plus PTX of the newest, which the driver compiles on
first launch for anything newer still. Building all five costs about twice a
single-architecture build: 13.7s against 6.1s on the development desktop.

`CUDA_ARCHS=native` builds only for the local GPU and is the one to use while
iterating.

The toolkit is chosen to satisfy that request rather than the other way round.
sm_120 needs CUDA 12.8 or newer, and a host commonly has several toolkits with
`CUDA_PATH` left pointing at the oldest -- `/usr/local/cuda` follows
`update-alternatives`, which on the development desktop still resolves to 12.3.
The build therefore prefers `CUDA_PATH` whenever it can build what was asked
for, and otherwise takes the toolkit that can, warning that it did so:

    warning: using /usr/local/cuda-12.8 instead of /usr/local/cuda, which
    cannot build sm_120; set CUDA_ARCHS=native to build only for this
    machine's GPU

To pin a toolkit for real, point `CUDA_PATH` at a versioned directory
(`/usr/local/cuda-12.8`) rather than the symlink, or move the symlink with
`sudo update-alternatives --config cuda`.

At runtime the toolkit matters again, and `CUDA_PATH` does not govern it:
CubeCL compiles its kernels with NVRTC using the compute capability read from
the device, and cudarc resolves cuSOLVER symbols on load. Both come from
whatever `LD_LIBRARY_PATH` finds, which is why `scripts/select_cuda.sh` sets
both variables for the test recipes.

## Cargo Features

**ndt_cuda crate**:
| Feature | Description |
|---------|-------------|
| `cuda` | Enable CUDA backend (default) |
| `profiling` | Enable timing instrumentation (minimal overhead) |
| `debug-iteration` | Enable per-iteration data collection (adds overhead) |
| `debug-cov` | Enable GPU vs CPU covariance comparison |
| `debug-vpp` | Enable voxel-per-point distribution logging |
| `debug` | All debug features combined (`debug-iteration` + `debug-cov` + `debug-vpp` + `profiling`) |
| `test-verbose` | Enable verbose println output in tests |

**cuda_ffi crate**:
| Feature | Description |
|---------|-------------|
| `test-verbose` | Enable verbose println output in tests |

Enable features with: `cargo test --features test-verbose` or `cargo build --features profiling`

**Important**: Use `profiling` for performance measurement. The `debug` feature adds significant overhead from per-iteration data collection.

## ROS 2 Integration Notes

**EKF Subscription QoS**: Uses depth 100 (matching Autoware) to buffer messages during node initialization. With depth 1, early EKF messages are lost before `spin()` starts processing callbacks.

**Initial Pose**: Demo scripts always enable `user_defined_initial_pose` for reproducible testing. Without this, the EKF initializes to an unknown state. The default pose is set in `ndt_replay_simulation.launch.xml`.

**SmartPoseBuffer**: Rejects interpolation when target timestamp is before first pose (matches Autoware behavior). Does NOT use fallback to first pose.

**ndt_align_srv Service Type**: Uses `autoware_internal_localization_msgs/srv/PoseWithCovarianceStamped` to match Autoware's `pose_initializer`. Init mode (`user_defined_initial_pose=false`) now works with Monte Carlo pose estimation.

## Launch File Conventions

**Following Autoware 1.5.0**: The launch files follow Autoware 1.5.0 structure:

- General localization configs (EKF, pose initializer, etc.) come from `localization_config_path` (defaults to `$(find-pkg-share autoware_launch)/config/localization`)
- Pointcloud preprocessor configs are self-contained in the package via `$(find-pkg-share cuda_ndt_matcher_launch)/config/`

**Package config structure** (`cuda_ndt_matcher_launch/config/`):
```
config/
├── cuda_scan_matcher.param.yaml          # CUDA-specific NDT parameters
└── pointcloud_preprocessor/             # Preprocessor configs (self-contained)
    ├── crop_box_filter_measurement_range.param.yaml
    ├── voxel_grid_filter.param.yaml
    └── random_downsample_filter.param.yaml
```

Preprocessor configs are referenced via `$(find-pkg-share cuda_ndt_matcher_launch)/config/pointcloud_preprocessor/` in both `cuda_localization.launch.xml` and `autoware_localization.launch.xml`, allowing independent tuning from stock Autoware NDT settings.

**Local util.launch.xml**: We maintain a local copy of `tier4_localization_launch/launch/util/util.launch.xml` in `cuda_ndt_matcher_launch/launch/util/` to fix a ROS 2 component name collision issue:

- **Problem**: ROS 2 component containers silently fail to load a composable node if another node with the same base name already exists, even in a different namespace
- **Symptom**: `voxel_grid_downsample_filter` from perception loads first, then localization's attempt to load its own `voxel_grid_downsample_filter` silently fails
- **Solution**: Renamed localization's node to `localization_voxel_grid_downsample_filter` to avoid the collision

This affects both `cuda_localization.launch.xml` and `autoware_localization.launch.xml`.

**This section described the fix as applied while the rename was absent from
both copies of `util.launch.xml`** (found 2026-08-04, applied then). The
failure is completely silent — `play_launch` reports the composable node as
loaded, the crop box either side of it keeps running at 10 Hz, and only NDT's
own log gives it away, with `NDT align failed: No sensor points available`
followed by `NDT scan matcher disabled` forever. If a replay produces zero
published poses, check for the node before anything else:

```bash
ros2 node list | grep voxel_grid          # expect localization_voxel_grid_downsample_filter
ros2 topic hz /localization/util/downsample/pointcloud
```

## Coding Conventions

- **Logging**: Use `rclrs::log_*!` in `cuda_ndt_matcher`, `tracing::*!` in `ndt_cuda`
- **Transforms**: Use nalgebra for all rotation/quaternion math
- **Format strings**: Use named parameters: `println!("{e}")` not `println!("{}", e)`

### Euler angles: nalgebra and Autoware compose them in opposite orders

**Never call nalgebra's `euler_angles()` or `UnitQuaternion::from_euler_angles()`
to produce or consume a pose vector.** This crate's `[x, y, z, roll, pitch, yaw]`
is Autoware's convention and nalgebra's is the reverse. This has now caused the
same bug three times.

| | composition |
|---|---|
| a pose vector here, and everything that reads one | **R = Rx(roll) · Ry(pitch) · Rz(yaw)** |
| nalgebra `from_euler_angles` / `euler_angles` | **R = Rz(yaw) · Ry(pitch) · Rx(roll)** |

The consumers that assume XYZ are `pose_to_transform_matrix`, the GPU kernels'
angular derivatives (`j_ang`, `h_ang`), and `derivatives/cpu.rs`.

Use these instead, in `optimization::types`:

| need | use |
|---|---|
| isometry → pose vector | `isometry_to_pose_vector` |
| pose vector → isometry | `pose_vector_to_isometry` |
| angles → rotation | `rotation_from_pose_angles` |
| rotation → angles | `pose_angles_from_rotation` |
| **isometry → 4x4 for a kernel** | `isometry_to_transform_matrix` — best when you already hold an isometry; no angle extraction, so no gimbal-lock case |

**Why it keeps getting through.** The two conventions agree whenever at most one
angle is non-zero, and nothing in the type system separates them — both are
`(f64, f64, f64)`. Any fixture with yaw-only motion passes either way. So a test
for this must use roll, pitch **and** yaw all non-zero; `(0.05, -0.04, 3.06)` is
the COSS heading as actually driven and is the case the existing tests use.

**Two ways such a test passes while proving nothing**, both hit while writing the
current one:

- With the default config (`use_gpu: false`) the GPU and CPU scorers both fall
  back to the same CPU routine and agree for the wrong reason. Force `use_gpu`
  and assert `is_gpu_active()`.
- If the pose carries the source cloud off the map, both sides return 0 for lack
  of correspondences and agree again. Build the source by pulling the target back
  through the pose, and assert the fixture actually scores before comparing.

**Existing guards**, worth extending rather than duplicating:

- `derivatives/gpu.rs::transform_roundtrip_tests` — asserts
  `pose_to_transform_matrix ∘ isometry_to_pose_vector` reproduces
  `isometry_to_transform_matrix`.
- `ndt.rs::test_batch_and_single_nvtl_agree_at_nonzero_rpy` — asserts the batch
  GPU scorer agrees with the single-pose scorer.

**The three occurrences**, all found by a number being wrong rather than by a
test failing:

1. 2026-08-03/04 — GPU NVTL read ~1.45x high (2.79 where the truth was 1.92).
   The publish gate was recalibrated against the wrong number instead of the
   number being fixed. Recorded at the `converged_param_nearest_voxel_transformation_likelihood`
   comment in `cuda_ndt_matcher_launch/config/cuda_scan_matcher.param.yaml`.
2. Same period — the pose vector carried nalgebra's convention at the optimizer
   boundary while every consumer assumed Autoware's. Fixing it lifted NVTL from
   2.00 to 2.80 and cut per-frame correction from 0.070 m to 0.025 m.
3. 2026-08-30 — `evaluate_nvtl_batch` and
   `compute_per_point_scores_for_visualization` fed `euler_angles()` to
   `GpuScoringPipeline`. NVTL 2.821 against a true 3.138 over 294 frames.

## CubeCL Limitations

1. **No dynamic array indexing**: Use fully unrolled loops instead of `arr[i as usize]`
2. **Parameter count limit**: Kernels with >12 parameters fail; combine buffers
3. **No `as usize`**: Use explicit indices

## Jetson Platform Notes

**cudarc CUDA version feature**: The `cudarc` crate requires a CUDA version feature that matches the symbols available in Jetson's Tegra CUDA libraries. Jetson's libcuda.so and libcusolver.so are missing some symbols that desktop CUDA has:

| Symbol | Required by | Available on Jetson |
|--------|-------------|---------------------|
| `cuEventElapsedTime_v2` | `cuda-12080`+ | ❌ No |
| `cusolverDnXgeev` | `cuda-12060`+ | ❌ No |

A third symbol matters on the x86 side: `cusolverDnXlarft`, required by
`cuda-12050`+ and absent from the CUDA 12.3 the development host builds against.

**Solution**: both `cudarc` copies are pinned to `cuda-12030`, the floor that
satisfies every host. Verified on JetPack 6.2 (R36.4.4) with `nm -D`.

**Pinning the normal dependency is not enough, and this is the part that bites.**
`cubecl-cuda` gates its CUDA 12.8 tensormap use on `#[cfg(cuda_12080)]`, which
its *own build script* sets from `CUDA_VERSION` as read from its
**build-dependency** copy of `cudarc` — a different copy, carrying
`cuda-version-from-build-system` + `fallback-latest`. That copy runs
`nvcc --version` and **assumes CUDA 13.0 when nvcc is off PATH**, after which
`cubecl-cuda` compiles its 12.8 branch against a `cudarc` that does not declare
those symbols:

```
error[E0432]: unresolved imports `cudarc::driver::sys::CUtensorMapIm2ColWideMode`,
              `cudarc::driver::sys::cuTensorMapEncodeIm2colWide`
```

Cargo's v2 resolver keeps build-dependency features separate from normal ones, so
`src/ndt_cuda/Cargo.toml` declares `cudarc` in `[build-dependencies]` at the same
floor, with a `build.rs` that exists only to make cargo resolve it. Do not remove
either; without them the build depends on whether nvcc happens to be on PATH, and
that answer is cached because PATH is not in cargo's fingerprint.

Raising the pin to `cuda-12080` "fixes" the build and is wrong: it declares
`cuEventElapsedTime_v2`, and `dynamic-loading` resolves every declared symbol in
`Lib::from_library()` with `.expect()`, so the node panics at startup on the Orin
instead of failing to compile.

## Key Files

| File | Purpose |
|------|---------|
| `ndt_cuda/src/optimization/full_gpu_pipeline_v2.rs` | Full GPU Newton with line search (graph kernels) |
| `ndt_cuda/src/optimization/debug.rs` | Per-iteration debug data structures |
| `cuda_ffi/csrc/ndt_graph_kernels.cu` | CUDA graph kernels (K1-K5) - Phase 24 |
| `cuda_ffi/csrc/ndt_graph_common.cuh` | Buffer layouts and configuration for graph kernels |
| `cuda_ffi/src/graph_ndt.rs` | Rust FFI bindings for graph kernels |
| `cuda_ffi/csrc/persistent_ndt.cu` | Legacy: CUDA persistent kernel with cooperative groups |
| `ndt_cuda/src/voxel_grid/gpu/pipeline.rs` | Zero-copy voxel grid construction |
| `cuda_ndt_matcher/src/main.rs` | ROS node entry point |

## Claude Code Practices

- Use `timeout` parameter on Bash tool instead of `timeout` command
- Use `run_in_background: true` for long-running processes
- Create temp files in `$project/tmp/` not `/tmp/`
- Always use Write/Edit tools to create files, not `cat << EOF` heredoc patterns in Bash
- **Do NOT modify files in `external/autoware_repo`** - copy to `src/` first

### Process Cleanup

When stopping `play_launch` or multi-process ROS systems, **kill the entire process group** to avoid orphan child processes that may interfere with topics:

```bash
# Get the PGID of play_launch and kill the whole group
PGID=$(ps -o pgid= -p $(pgrep -f play_launch) | tr -d ' ')
kill -9 -$PGID

# Or use pkill with -g flag to kill process group
pkill -9 -g $PGID
```

**Never** use `pkill -9 -f play_launch` alone as it leaves orphaned child processes (component containers, ros2 nodes) that hold topics and prevent clean restarts.

**Common orphan processes to kill:**
```bash
pkill -9 -f "component_container"      # ROS 2 composable node containers
pkill -9 -f "component_container_mt"   # Multi-threaded containers
pkill -9 -f "robot_state_publisher"    # TF publisher
pkill -9 -f "ros2-daemon"              # ROS 2 CLI daemon
```

## Profiling

**Release profiling** (minimal overhead, for accurate performance comparison):
```bash
just profile-compare           # Full workflow: build, run both, compare
just profile-quick             # Quick comparison (release builds only)
just run-cuda-profiling        # CUDA with timing data only
just run-builtin-profiling     # Autoware with timing data only
just compare-profiling         # Analyze results
```

**Init pose profiling** (Monte Carlo pose initialization):
```bash
just profile-init              # Full workflow: run both with init mode, compare timing and poses
just run-cuda-init             # CUDA with init mode
just run-builtin-init          # Autoware with init mode
```

**Resource profiling** (CPU, GPU, power on Jetson):
```bash
just profile-resource          # Full workflow with tegrastats monitoring
just analyze-resource          # Compare per-node CPU, memory, GPU, power
just analyze-system-stats      # Compare total system CPU and memory
just analyze-tegrastats-cpu    # Analyze tegrastats CPU data
```

**Pose comparison** (from rosbag recordings):
```bash
just compare-init-poses-latest              # Compare poses from latest rosbags
just compare-init-poses <cuda> <autoware>   # Compare specific rosbags
just compare-poses                          # Compare tracking poses from profiling logs
```

**Debug profiling** (full debug data, adds overhead):
```bash
just run-cuda-debug            # CUDA with all debug features
just run-builtin-debug         # Autoware with all debug features
```

**Build differences**:
| Build | Feature | Overhead | Use Case |
|-------|---------|----------|----------|
| `build-cuda-profiling` | `profiling` | Minimal | Performance measurement |
| `build-cuda-debug` | `debug` | Significant | Debug data collection |

Output files (in `logs/profiling/<date>/`):
- `ndt_cuda_profiling.jsonl` - CUDA timing data
- `ndt_autoware_profiling.jsonl` - Autoware timing data
- `init_pose_comparison.txt` - Init pose comparison results (from `profile-init`)
- `cuda_tegrastats.log` / `autoware_tegrastats.log` - Jetson tegrastats logs
- `cuda_system_stats.csv` / `autoware_system_stats.csv` - System-wide CPU/memory
- `cuda_metrics.csv` / `autoware_metrics.csv` - Per-node metrics from play_launch

Analysis scripts:
- `scripts/profile_ndt_comparison.py` - Compare CUDA vs Autoware performance
- `scripts/compare_init_poses.py` - Compare init poses from rosbags
- `scripts/compare_poses.py` - Compare tracking poses from profiling logs
- `scripts/analyze_profile.py` - Analyze profile directory structure
- `scripts/analyze_resource_usage.py` - Compare CPU, GPU, power from tegrastats
- `scripts/analyze_system_stats.py` - Compare total system CPU/memory
- `scripts/analyze_tegrastats_cpu.py` - Parse tegrastats for per-core CPU

## Comparison Testing

The `tests/comparison/` directory contains a fork of `autoware_ndt_scan_matcher` with debug patches for comparison testing. Builtin NDT recipes delegate to `tests/comparison/justfile`.

**Setup:**
```bash
# Initialize submodule (if not already done)
git submodule update --init tests/comparison/autoware_universe

# Build patched Autoware NDT
just build-comparison
```

**Usage:**
```bash
# Run Autoware (unpatched, no debug)
just run-builtin

# Run Autoware with debug output (requires build-comparison first)
just run-builtin-debug

# Dump voxel data for comparison
just dump-voxels-cuda
just dump-voxels-autoware

# Full voxel comparison workflow (dump both + compare)
just compare-voxels
```

**Architecture:**
- `tests/comparison/autoware_universe/` - git submodule with debug patches
- `tests/comparison/justfile` - builds and runs patched Autoware
- Main justfile delegates: `run-builtin`, `run-builtin-debug`, `dump-voxels-autoware`, `analyze-debug-autoware`
- `run_ndt_simulation.sh` overlays `tests/comparison/install/` when available

**Patches included:**
- Per-iteration debug output (score, gradient, Hessian)
- Voxel grid dump for covariance comparison
- Convergence status logging

## Validation Status

### Covariance Formula Bug Fixed (2026-01-19)

**Root cause found and fixed**: The CPU voxel grid construction in `types.rs` used an incorrect covariance formula.

**Bug location**: `src/ndt_cuda/src/voxel_grid/types.rs:69-82`

**Wrong formula** (caused ~73% score):
```rust
cov = (sum_sq/n - mean*mean^T) * (n-1)/n  // WRONG
```

**Correct formula** (matches Autoware):
```rust
cov = (sum_sq - n*mean*mean^T) / (n-1)    // Standard sample covariance
```

**Impact**: The ratio between wrong/correct formulas is `(n-1)²/n²`:
| Points (n) | Formula ratio | Observed in dumps |
|------------|---------------|-------------------|
| 6          | 0.69          | 0.55 (matched)    |
| 10         | 0.81          | 0.63 (matched)    |
| 100        | 0.98          | 0.94 (matched)    |

**Verification**:
- GPU vs CPU cov_sums comparison: ratio = 1.000000 (exact match)
- All 417 unit tests pass (351 ndt_cuda + 66 cuda_ffi)
- All 7 Autoware comparison tests pass

**Note**: The GPU pipeline in `statistics.rs` was already correct (accumulates centered deviations and divides by n-1). Only the CPU path in `types.rs::from_statistics` had the bug.

**Investigation tools** (for future debugging):
```bash
# Generate voxel dumps
NDT_DUMP_VOXELS=1 just run-cuda
NDT_DUMP_VOXELS=1 just run-builtin

# Compare voxels
python3 tmp/compare_matching_voxels.py
python3 tmp/analyze_by_point_count.py

# Debug GPU vs CPU cov_sums
NDT_DEBUG_COV=1 cargo test -p ndt_cuda -- voxel --nocapture
```

### Autoware Identity Initialization Fix (2026-01-31)

**Root cause**: Autoware initializes `leaf.cov_` to Identity matrix, not zero.

**Bug location**: `multi_voxel_grid_covariance_omp.h:132`
```cpp
cov_(Eigen::Matrix3d::Identity()),  // Autoware initializes to I, not 0
```

**Impact**: When accumulating `leaf.cov_ += pt * pt.T`, Autoware starts from I, giving:
```cpp
cov = (I + Σ(x*xᵀ) - n*mean*meanᵀ) / (n-1)
    = standard_cov + I/(n-1)
```

This adds `1/(n-1)` to each diagonal element. For n=6, this is +0.2 per diagonal.

**Fix applied to**:
- `src/ndt_cuda/src/voxel_grid/types.rs:132`: Added `+ Matrix3::identity()` to numerator
- `src/ndt_cuda/src/voxel_grid/gpu/statistics.rs:437-439`: Added `+ denom` to diagonal elements
- `src/ndt_cuda/src/voxel_grid/gpu/autoware_comparison.rs:148`: Updated test helper

**Verification**: All 421 tests pass (355 ndt_cuda + 66 cuda_ffi)

## Performance Summary (2026-01-30)

See `docs/performance/autoware-comparison.md` for detailed analysis.

### Tracking Performance

| Platform | CUDA | Autoware | Speedup |
|----------|------|----------|---------|
| Desktop x86 (RTX 5090) | 199.7 Hz | 125.4 Hz | **1.59x** |
| Jetson AGX Orin (64GB) | 34.8 Hz | 26.5 Hz | **1.32x** |

### Init Pose Performance

| Platform | CUDA | Autoware | Speedup |
|----------|------|----------|---------|
| Desktop x86 | 2.6s | 6.7s | **2.57x** |
| Jetson AGX Orin | 7.4s | 22.6s | **3.05x** |

### Resource Efficiency (Jetson)

| Metric | CUDA | Autoware | Improvement |
|--------|------|----------|-------------|
| NDT CPU usage | 34.8% | 81.0% | **57% less** |
| System CPU (12 cores) | 520.7% | 570.7% | 0.5 cores freed |
| GPU utilization | 11% avg | 0.2% | Headroom for perception |
| Power consumption | 5505 mW | 5468 mW | Equal |
| Throughput/Watt | - | - | **30% better** |

**Key insight**: NDT is the CPU bottleneck (46% of all Autoware CPU). CUDA offloads this to GPU with minimal power increase.
