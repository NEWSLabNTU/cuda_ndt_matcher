# CubeCL NDT Implementation Roadmap

## Status

| Phase | Description | Status | Location |
|-------|-------------|--------|----------|
| 1 | Voxel Grid Construction | ✅ Complete | [archive](../archive/phase-1-voxel-grid.md) |
| 2 | Derivative Computation | ✅ Complete | [archive](../archive/phase-2-derivatives.md) |
| 3 | Newton Optimization | ✅ Complete | [archive](../archive/phase-3-newton.md) |
| 4 | Scoring & NVTL | ✅ Complete | [archive](../archive/phase-4-scoring.md) |
| 5 | Integration & Optimization | ✅ Complete | [archive](../archive/phase-5-integration.md) |
| 6 | Validation & Production | ⚠️ Partial | [phase-6-validation.md](phase-6-validation.md) |
| 7 | ROS Integration & Features | ✅ Complete | [archive](../archive/phase-7-ros-features.md) |
| 8 | Autoware Parity | ✅ Complete | [archive](../archive/phase-8-autoware-parity.md) |
| 9 | Full GPU Acceleration | ⚠️ Partial | [phase-9-gpu-acceleration.md](phase-9-gpu-acceleration.md) |
| 10 | SmartPoseBuffer | ✅ Complete | [archive](../archive/phase-10-smart-pose-buffer.md) |
| 11 | GPU Zero-Copy Voxel Pipeline | ✅ Complete | [archive](../archive/phase-11-gpu-zero-copy-pipeline.md) |
| 12 | GPU Derivative Pipeline | ⚠️ Superseded | [archive](../archive/phase-12-gpu-derivative-pipeline.md) |
| 13 | GPU Scoring Pipeline | ✅ Complete | [archive](../archive/phase-13-gpu-scoring-pipeline.md) |
| 14 | Full GPU Newton | ✅ Complete | [archive](../archive/phase-14-iteration-optimization.md) |
| 15 | Full GPU + Line Search | ✅ Complete | [archive](../archive/phase-15-gpu-line-search.md) |
| 16 | GPU Initial Pose Pipeline | ✅ Complete | [archive](../archive/phase-16-gpu-initial-pose-pipeline.md) |
| 17 | Kernel Optimization | ✅ Complete | [archive](../archive/phase-17-kernel-optimization.md) |
| 18 | Persistent Kernel Features | ✅ Complete | [archive](../archive/phase-18-persistent-kernel-features.md) |
| 19 | Cleanup & Enhancements | ✅ Complete | [archive](../archive/phase-19-cleanup.md) |
| 22 | Batch Multi-Alignment | ✅ Complete | [archive](../archive/phase-22-batch-alignment.md) |
| 23 | GPU Utilization | ⚠️ Partial | [phase-23-gpu-utilization.md](phase-23-gpu-utilization.md) |
| 24 | CUDA Graphs Pipeline | ✅ Complete | [archive](../archive/phase-24-cuda-graphs-pipeline.md) |
| 25 | Code Restructure & Quality | ✅ Complete | [archive](../archive/phase-25-code-restructure.md) |
| 26 | Nextest Migration | ✅ Complete | [archive](../archive/phase-26-nextest-migration.md) |
| 27 | CPU Parallelism | ✅ Complete | [archive](../archive/phase-27-cpu-parallelism.md) |

## Active Phases

- **Phase 6**: Algorithm verified; rosbag testing pending
- **Phase 9**: 9.1 workaround complete, 9.2 GPU voxel grid complete; remaining items pending
- **Phase 23**: 23.1 async streams complete; texture/warp pending

## Other Documents

- [Implementation Notes](implementation-notes.md) — Dependencies, risks, references
- [Debug Buffer Design](../archive/phase-19-4-debug-buffer-design.md) — Per-iteration debug data layout
