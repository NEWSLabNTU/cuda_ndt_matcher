# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

CUDA/Rust re-implementation of Autoware's `ndt_scan_matcher` using CubeCL for GPU compute.

**Reference implementation**: `external/autoware_core/localization/autoware_ndt_scan_matcher/`

**Documentation**:
- `docs/autoware-comparison.md` - Feature comparison and GPU acceleration status
- `docs/roadmap/` - Implementation phases and status
- `docs/profiling-results.md` - Performance analysis
- `docs/optimization-approaches.md` - Potential optimizations for iteration reduction

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

**Pipeline config**: `PipelineV2Config::enable_debug = true` collects per-iteration debug data (score, gradient, Hessian, step size) from the persistent kernel with zero overhead when disabled.

## Cargo Features

**ndt_cuda crate**:
| Feature | Description |
|---------|-------------|
| `cuda` | Enable CUDA backend (default) |
| `profiling` | Enable detailed timing instrumentation and hash table debug logs |
| `test-verbose` | Enable verbose println output in tests |

**cuda_ffi crate**:
| Feature | Description |
|---------|-------------|
| `test-verbose` | Enable verbose println output in tests |

Enable features with: `cargo test --features test-verbose` or `cargo build --features profiling`

## ROS 2 Integration Notes

**EKF Subscription QoS**: Uses depth 100 (matching Autoware) to buffer messages during node initialization. With depth 1, early EKF messages are lost before `spin()` starts processing callbacks.

**Initial Pose**: Demo scripts always enable `user_defined_initial_pose` for reproducible testing. Without this, the EKF initializes to an unknown state. The default pose is set in `ndt_replay_simulation.launch.xml`.

**SmartPoseBuffer**: Rejects interpolation when target timestamp is before first pose (matches Autoware behavior). Does NOT use fallback to first pose.

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
| `ndt_cuda/src/optimization/full_gpu_pipeline_v2.rs` | Full GPU Newton with line search (persistent kernel) |
| `ndt_cuda/src/optimization/debug.rs` | Per-iteration debug data structures |
| `cuda_ffi/csrc/persistent_ndt.cu` | CUDA persistent kernel with cooperative groups |
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
