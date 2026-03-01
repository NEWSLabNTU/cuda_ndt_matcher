# Architecture

## ROS Interface (Autoware-compatible)

### Subscriptions

| Topic                                 | Type                      | Description           |
|---------------------------------------|---------------------------|-----------------------|
| `points_raw`                          | PointCloud2               | Sensor point cloud    |
| `ekf_pose_with_covariance`            | PoseWithCovarianceStamped | Initial pose from EKF |
| `regularization_pose_with_covariance` | PoseWithCovarianceStamped | GNSS regularization   |

### Publishers

| Topic                      | Type                      | Description          |
|----------------------------|---------------------------|----------------------|
| `ndt_pose`                 | PoseStamped               | Estimated pose       |
| `ndt_pose_with_covariance` | PoseWithCovarianceStamped | Pose with covariance |

### Services

| Service              | Type    | Description         |
|----------------------|---------|---------------------|
| `trigger_node_srv`   | SetBool | Enable/disable node |
| `pcd_loader_service` | -       | Map loading client  |

### Parameters

```yaml
frame:
  base_frame: "base_link"
  ndt_base_frame: "ndt_base_link"
  map_frame: "map"

ndt:
  resolution: 2.0
  max_iterations: 30
  trans_epsilon: 0.01
  step_size: 0.1
  num_threads: 4

initial_pose_estimation:
  particles_num: 200
  n_startup_trials: 100

covariance:
  covariance_estimation_type: 0  # 0=FIXED, 1=LAPLACE, 2=MULTI_NDT, 3=MULTI_NDT_SCORE
```

## System Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    ROS 2 Node (rclrs)                   │
├─────────────────────────────────────────────────────────┤
│  Inputs:                    │  Outputs:                 │
│  - points_raw               │  - ndt_pose               │
│  - ekf_pose_with_covariance │  - ndt_pose_with_covariance │
│  - regularization_pose      │  - diagnostics            │
└──────────────┬──────────────┴───────────────────────────┘
               │
       ┌───────▼───────┐
       │  NDT Manager  │
       └───────┬───────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│fast_gicp│ │ cubecl │ │  Rust  │
│  CUDA   │ │kernels │ │  CPU   │
└────────┘ └────────┘ └────────┘
```

## Module Responsibilities

### fast_gicp_rust

- NDT registration: `NDTCuda::align()`
- Voxelization: GPU voxel grid
- Score computation: transformation probability

### cubecl

- Multi-NDT covariance estimation
- Laplace approximation (Hessian-based)
- Monte Carlo initial pose sampling

### Rust CPU

- Map management and caching
- Pose interpolation
- GNSS regularization
- Diagnostics and validation
