# CUDA NDT Matcher

Re-implementation of Autoware's `ndt_scan_matcher` with CUDA acceleration.

## Goals

1. GPU-accelerated NDT scan matching for real-time localization
2. Rust implementation with ROS 2 integration
3. API-compatible with Autoware ndt_scan_matcher

## References

- `external/autoware_core/localization/autoware_ndt_scan_matcher/` - Autoware C++ reference
- `external/fast_gicp_rust/` - Rust bindings for fast_gicp CUDA NDT (can be improved)

## Components

| Component               | Implementation | Notes                       |
|-------------------------|----------------|-----------------------------|
| NDT Core                | fast_gicp_rust | Tested CUDA kernels         |
| Voxel Grid              | fast_gicp_rust | GPU voxelization            |
| Covariance Estimation   | cubecl         | Multi-NDT, Laplace approx   |
| Initial Pose Estimation | cubecl         | Monte Carlo particles       |
| Dynamic Map Loading     | Rust           | CPU-side map management     |
| ROS Node                | rclrs          | Compatible with Autoware    |

## Workflow

```
fast_gicp_rust (existing, can be extended)
├── NDTCuda           # Core registration
├── FastVGICPCuda     # Alternative algorithm
└── Voxel operations  # GPU voxelization

cubecl (new kernels)
├── Covariance estimation
├── Monte Carlo initial pose
└── Score computation variants

ROS Node (Autoware-compatible interface)
├── Topics: same as ndt_scan_matcher
├── Services: same as ndt_scan_matcher
└── Parameters: same as ndt_scan_matcher
```

## fast_gicp_rust Improvements

Potential enhancements to `external/fast_gicp_rust/`:
- Bug fixes discovered during integration
- Additional configuration options
- Performance optimizations
- New algorithm variants
