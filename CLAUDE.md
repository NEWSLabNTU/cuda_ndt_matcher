# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

CUDA/Rust re-implementation of Autoware's `ndt_scan_matcher` using CubeCL for GPU compute.

**Reference implementation**: `external/autoware_core/localization/autoware_ndt_scan_matcher/`

**Documentation**:
- `docs/autoware-comparison.md` - Feature comparison and GPU acceleration status
- `docs/roadmap/` - Implementation phases and status
- `docs/profiling-results.md` - Performance analysis

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
cargo --config build/ros2_cargo_config.toml test -p ndt_cuda --lib test_name
```

## Running

```bash
# Demo mode with logging
just run-cuda      # CUDA NDT
just run-builtin   # Autoware NDT (baseline)

# Service mode
just start-ndt-cuda && just start-rosbag && just enable-ndt
just log-ndt-cuda  # Monitor output
just stop-ndt-cuda
```

See `docs/rosbag-replay-guide.md` for custom rosbag testing.

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

## Coding Conventions

- **Logging**: Use `rclrs::log_*!` in `cuda_ndt_matcher`, `tracing::*!` in `ndt_cuda`
- **Transforms**: Use nalgebra for all rotation/quaternion math
- **Format strings**: Use named parameters: `println!("{e}")` not `println!("{}", e)`

## CubeCL Limitations

1. **No dynamic array indexing**: Use fully unrolled loops instead of `arr[i as usize]`
2. **Parameter count limit**: Kernels with >12 parameters fail; combine buffers
3. **No `as usize`**: Use explicit indices

## Key Files

| File | Purpose |
|------|---------|
| `ndt_cuda/src/optimization/full_gpu_pipeline_v2.rs` | Full GPU Newton with line search |
| `ndt_cuda/src/optimization/gpu_pipeline_kernels.rs` | All GPU kernels |
| `ndt_cuda/src/voxel_grid/gpu/pipeline.rs` | Zero-copy voxel grid construction |
| `cuda_ndt_matcher/src/main.rs` | ROS node entry point |

## Claude Code Practices

- Use `timeout` parameter on Bash tool instead of `timeout` command
- Use `run_in_background: true` for long-running processes
- Create temp files in `$project/tmp/` not `/tmp/`
- **Do NOT modify files in `external/autoware_repo`** - copy to `src/` first
