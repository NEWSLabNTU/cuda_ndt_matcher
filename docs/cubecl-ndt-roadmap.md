# CubeCL NDT Implementation Roadmap

This document outlines the plan to implement custom CUDA kernels for NDT scan matching using [CubeCL](https://github.com/tracel-ai/cubecl), phasing out the fast-gicp dependency.

## Current Status (2025-12-31)

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Voxel Grid | ✅ Complete | CPU implementation with KD-tree radius search |
| Phase 2: Derivatives | ✅ Complete | CPU multi-voxel matching, GPU kernels defined |
| Phase 3: Newton Solver | ✅ Complete | More-Thuente line search implemented |
| Phase 4: Scoring | ✅ Complete | NVTL and transform probability |
| Phase 5: Integration | ✅ Complete | API complete, GPU runtime implemented |
| Phase 6: Validation | ⚠️ Partial | Algorithm verified, rosbag testing pending |
| Phase 7: ROS Features | ⚠️ Mostly Complete | TF, map loading, multi-NDT, Monte Carlo viz; GPU scoring not integrated |

**Core NDT algorithm is fully implemented on CPU and matches Autoware's pclomp.**
**GPU runtime is implemented with CubeCL for accelerated scoring (requires CUDA).**

## Background

### Why Replace fast-gicp?

The fast-gicp NDTCuda implementation has fundamental issues:

| Issue | Impact |
|-------|--------|
| Uses Levenberg-Marquardt optimizer | Never converges properly (hits 30 iterations every time) |
| Different cost function (Mahalanobis) | Different optimization landscape vs pclomp |
| SO(3) exponential map parameterization | Different Jacobian structure than Euler angles |
| No exposed iteration count | Cannot diagnose convergence |

### Why CubeCL?

- **Pure Rust**: No C++/CUDA FFI complexity
- **Multi-platform**: CUDA, ROCm, WebGPU from same codebase
- **Type-safe**: Rust's type system for GPU code
- **Automatic vectorization**: `Line<T>` handles SIMD
- **Autotuning**: Runtime optimization of kernel parameters

## Architecture Overview

```
cuda_ndt_matcher/
├── src/
│   ├── cuda_ndt_matcher/           # ROS node (existing)
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── ndt_manager.rs      # Will use new NDT engine
│   │   │   └── ...
│   │   └── Cargo.toml
│   │
│   └── ndt_cuda/                   # NEW: CubeCL NDT library
│       ├── src/
│       │   ├── lib.rs
│       │   ├── voxel_grid/         # Phase 1: Voxelization
│       │   │   ├── mod.rs
│       │   │   ├── kernels.rs      # CubeCL kernels
│       │   │   └── types.rs
│       │   ├── derivatives/        # Phase 2: Derivative computation
│       │   │   ├── mod.rs
│       │   │   ├── jacobian.rs
│       │   │   ├── hessian.rs
│       │   │   └── kernels.rs
│       │   ├── optimization/       # Phase 3: Newton solver
│       │   │   ├── mod.rs
│       │   │   ├── newton.rs
│       │   │   └── line_search.rs
│       │   ├── scoring/            # Phase 4: Probability scoring
│       │   │   ├── mod.rs
│       │   │   └── kernels.rs
│       │   └── ndt.rs              # High-level API
│       ├── benches/
│       └── Cargo.toml
```

## Phase 1: Voxel Grid Construction (2-3 weeks)

### Goal
Build GPU-accelerated voxel grid from point cloud map with covariance computation.

### Components

#### 1.1 Voxel ID Computation
```rust
#[cube(launch_unchecked)]
fn compute_voxel_ids<F: Float>(
    points: &Array<Line<F>>,      // [N, 3] point cloud
    voxel_ids: &mut Array<u32>,   // [N] output voxel IDs
    min_bound: &Array<F>,         // [3] bounding box min
    resolution: F,
    grid_dims: &Array<u32>,       // [3] grid dimensions
) {
    let idx = ABSOLUTE_POS;
    if idx >= points.len() / 3 {
        return;
    }

    let x = (points[idx * 3 + 0] - min_bound[0]) / resolution;
    let y = (points[idx * 3 + 1] - min_bound[1]) / resolution;
    let z = (points[idx * 3 + 2] - min_bound[2]) / resolution;

    let ix = Line::cast_from(x);
    let iy = Line::cast_from(y);
    let iz = Line::cast_from(z);

    voxel_ids[idx] = ix + iy * grid_dims[0] + iz * grid_dims[0] * grid_dims[1];
}
```

#### 1.2 Point Accumulation (Parallel Histogram)
- Sort points by voxel ID (use `cub::DeviceRadixSort` via cuBLAS or implement in CubeCL)
- Segment reduce to compute per-voxel statistics

#### 1.3 Covariance Computation
```rust
#[cube(launch_unchecked)]
fn compute_covariance<F: Float>(
    // Per-voxel accumulated statistics
    point_sums: &Array<F>,        // [V, 3] sum of points
    point_sq_sums: &Array<F>,     // [V, 6] sum of x*x^T (upper triangle)
    point_counts: &Array<u32>,    // [V] count per voxel

    // Output
    means: &mut Array<F>,         // [V, 3] voxel centroids
    covariances: &mut Array<F>,   // [V, 9] 3x3 covariance matrices
    inv_covariances: &mut Array<F>, // [V, 9] inverse covariances
) {
    let voxel_idx = ABSOLUTE_POS;
    let count = point_counts[voxel_idx];

    if count < 6 {  // Minimum points for valid covariance
        return;
    }

    // Compute mean
    let mean_x = point_sums[voxel_idx * 3 + 0] / F::cast_from(count);
    // ... mean_y, mean_z

    // Compute covariance using single-pass formula:
    // cov = (sum_sq - sum * mean^T) / (n - 1)
    // ...

    // Regularize via eigendecomposition (or simplified approach)
    // ...

    // Compute inverse
    // ...
}
```

#### 1.4 Eigenvalue Regularization
For singular covariances, inflate small eigenvalues:
- Option A: Use cuSOLVER for batched eigendecomposition
- Option B: Implement power iteration in CubeCL (simpler, may be sufficient)
- Option C: Use Jacobi eigenvalue algorithm (pure CubeCL)

### Data Structures
```rust
/// GPU voxel grid
pub struct GpuVoxelGrid {
    /// Voxel centroids [V, 3]
    pub means: CubeBuffer<f32>,
    /// Inverse covariances [V, 9] (row-major 3x3)
    pub inv_covariances: CubeBuffer<f32>,
    /// Point counts per voxel
    pub counts: CubeBuffer<u32>,
    /// Voxel coordinates for spatial lookup
    pub coords: CubeBuffer<i32>,  // [V, 3]
    /// Hash table for O(1) voxel lookup
    pub hash_table: GpuHashTable,
    /// Grid parameters
    pub resolution: f32,
    pub min_bound: [f32; 3],
}
```

### Tests
- [x] Voxel ID matches CPU implementation
- [x] Mean/covariance matches within floating-point tolerance
- [x] Inverse covariance is valid (no NaN/Inf)
- [x] KD-tree radius search returns correct voxels
- [x] Multi-voxel correspondences match Autoware behavior
- [ ] GPU kernel performance: < 10ms for 100K point cloud

---

## Phase 2: Derivative Computation (3-4 weeks)

### Goal
Compute gradient (6x1) and Hessian (6x6) for Newton optimization.

This is the **most compute-intensive** part, parallelized over input points.

### Components

#### 2.1 Angular Derivatives (Precompute)
Compute j_ang matrices from current rotation estimate (Euler angles):
```rust
/// Precomputed angular derivatives for current pose
pub struct AngularDerivatives {
    /// 8x4 Jacobian components (Eq. 6.19)
    pub j_ang: [[f32; 4]; 8],
    /// 16x4 Hessian components (Eq. 6.21)
    pub h_ang: [[f32; 4]; 16],
}

fn compute_angular_derivatives(roll: f32, pitch: f32, yaw: f32) -> AngularDerivatives {
    let (sx, cx) = (roll.sin(), roll.cos());
    let (sy, cy) = (pitch.sin(), pitch.cos());
    let (sz, cz) = (yaw.sin(), yaw.cos());

    // Magnusson 2009, Eq. 6.19
    let j_ang = [
        [-sx*sz + cx*sy*cz, -sx*cz - cx*sy*sz, -cx*cy, 0.0],
        [cx*sz + sx*sy*cz, cx*cz - sx*sy*sz, -sx*cy, 0.0],
        // ... remaining 6 rows
    ];
    // ...
}
```

#### 2.2 Point Jacobian Kernel
For each input point, compute 4x6 Jacobian:
```rust
#[cube(launch_unchecked)]
fn compute_point_jacobians<F: Float>(
    points: &Array<F>,            // [N, 3] source points
    j_ang: &Array<F>,             // [8, 4] precomputed angular derivatives
    jacobians: &mut Array<F>,     // [N, 4, 6] output Jacobians
) {
    let idx = ABSOLUTE_POS;
    if idx >= points.len() / 3 {
        return;
    }

    let x = points[idx * 3 + 0];
    let y = points[idx * 3 + 1];
    let z = points[idx * 3 + 2];

    // Eq. 6.18: point_gradient = [I_3x3 | d(Rx)/d(r,p,y)]
    // First 3 columns: identity for translation
    jacobians[idx * 24 + 0] = F::new(1.0);  // dx/dtx
    jacobians[idx * 24 + 7] = F::new(1.0);  // dy/dty
    jacobians[idx * 24 + 14] = F::new(1.0); // dz/dtz

    // Last 3 columns: rotation derivatives
    // j_ang[0] * x + j_ang[1] * y + j_ang[2] * z
    // ...
}
```

#### 2.3 Voxel Correspondence & Score Accumulation
The critical kernel: find neighboring voxels and accumulate gradient/Hessian.

```rust
#[cube(launch_unchecked)]
fn compute_derivatives<F: Float>(
    // Input
    transformed_points: &Array<F>,  // [N, 3] T(p) * source_points
    original_points: &Array<F>,     // [N, 3] source points (for Jacobian)
    jacobians: &Array<F>,           // [N, 4, 6] point Jacobians

    // Voxel grid (target)
    voxel_means: &Array<F>,         // [V, 3]
    voxel_inv_covs: &Array<F>,      // [V, 9]
    voxel_hash: &Array<u32>,        // Hash table for lookup

    // Gaussian parameters
    gauss_d1: F,
    gauss_d2: F,
    resolution: F,

    // Output (per-thread accumulation)
    scores: &mut Array<F>,          // [num_blocks]
    gradients: &mut Array<F>,       // [num_blocks, 6]
    hessians: &mut Array<F>,        // [num_blocks, 36]
) {
    let idx = ABSOLUTE_POS;

    // 1. Get transformed point
    let x_trans = [
        transformed_points[idx * 3 + 0],
        transformed_points[idx * 3 + 1],
        transformed_points[idx * 3 + 2],
    ];

    // 2. Find neighboring voxels (DIRECT7: center + 6 neighbors)
    let voxel_coord = compute_voxel_coord(x_trans, resolution);

    for offset in NEIGHBOR_OFFSETS {
        let neighbor = voxel_coord + offset;
        let voxel_idx = hash_lookup(voxel_hash, neighbor);

        if voxel_idx < 0 {
            continue;
        }

        // 3. Compute residual
        let mean = load_vec3(voxel_means, voxel_idx);
        let x_diff = x_trans - mean;

        // 4. Compute Mahalanobis distance
        let inv_cov = load_mat3x3(voxel_inv_covs, voxel_idx);
        let mahal = dot(x_diff, mat_vec_mul(inv_cov, x_diff));

        // 5. Compute score (Eq. 6.9)
        let e_x_cov_x = F::exp(-gauss_d2 * mahal / F::new(2.0));
        let score_inc = -gauss_d1 * e_x_cov_x;

        // 6. Accumulate gradient (Eq. 6.12)
        // gradient += gauss_d1 * gauss_d2 * e_x_cov_x * J^T * inv_cov * x_diff

        // 7. Accumulate Hessian (Eq. 6.13)
        // Complex: involves point_hessian and second derivatives

        // Use atomic add for thread-safe accumulation
        atomic_add(&scores[CUBE_POS], score_inc);
        // ...
    }
}
```

#### 2.4 Block-Level Reduction
Reduce per-block accumulators to final gradient/Hessian:
```rust
#[cube(launch_unchecked)]
fn reduce_derivatives<F: Float>(
    block_scores: &Array<F>,      // [num_blocks]
    block_gradients: &Array<F>,   // [num_blocks, 6]
    block_hessians: &Array<F>,    // [num_blocks, 36]

    total_score: &mut Array<F>,   // [1]
    total_gradient: &mut Array<F>, // [6]
    total_hessian: &mut Array<F>,  // [36]
) {
    // Parallel reduction
    // ...
}
```

### Tests
- [x] Gradient matches CPU within 1e-5 (verified with finite difference test)
- [x] Hessian matches CPU within 1e-4 (verified symmetry test)
- [x] Score matches CPU within 1e-6
- [x] Gaussian parameters (d1, d2, d3) match Autoware exactly
- [x] Multi-voxel radius search accumulates contributions correctly
- [ ] GPU kernel performance: < 5ms for 50K source points

---

## Phase 3: Newton Optimization (2 weeks)

### Goal
Implement Newton's method with optional More-Thuente line search.

### Components

#### 3.1 Newton Step (CPU)
The 6x6 linear solve is too small for GPU benefit:
```rust
pub fn newton_step(
    gradient: &Vector6<f64>,
    hessian: &Matrix6<f64>,
) -> Vector6<f64> {
    // SVD solve: delta = -H^{-1} * g
    let svd = hessian.svd(true, true);
    svd.solve(gradient, 1e-10).unwrap() * -1.0
}
```

#### 3.2 Transformation Update (GPU)
Apply delta to transformation and transform all points:
```rust
#[cube(launch_unchecked)]
fn transform_points<F: Float>(
    points: &Array<F>,           // [N, 3] source points
    transform: &Array<F>,        // [16] 4x4 transformation matrix
    output: &mut Array<F>,       // [N, 3] transformed points
) {
    let idx = ABSOLUTE_POS;
    if idx >= points.len() / 3 {
        return;
    }

    let x = points[idx * 3 + 0];
    let y = points[idx * 3 + 1];
    let z = points[idx * 3 + 2];

    // T * [x, y, z, 1]^T
    output[idx * 3 + 0] = transform[0]*x + transform[1]*y + transform[2]*z + transform[3];
    output[idx * 3 + 1] = transform[4]*x + transform[5]*y + transform[6]*z + transform[7];
    output[idx * 3 + 2] = transform[8]*x + transform[9]*y + transform[10]*z + transform[11];
}
```

#### 3.3 Convergence Check
```rust
pub fn check_convergence(
    delta: &Vector6<f64>,
    trans_epsilon: f64,
    iteration: usize,
    max_iterations: usize,
) -> bool {
    iteration >= max_iterations || delta.norm() < trans_epsilon
}
```

#### 3.4 Line Search (Optional)
More-Thuente line search - currently disabled in Autoware due to local minima issues.
Implement as optional feature for experimentation.

### Main Loop
```rust
pub fn align(
    &mut self,
    source: &GpuPointCloud,
    initial_guess: Isometry3<f64>,
) -> NdtResult {
    let mut transform = initial_guess;

    for iteration in 0..self.max_iterations {
        // 1. Transform source points (GPU)
        self.transform_points(source, &transform);

        // 2. Compute angular derivatives (CPU, tiny)
        let ang_deriv = compute_angular_derivatives(&transform);

        // 3. Compute point Jacobians (GPU)
        self.compute_point_jacobians(source, &ang_deriv);

        // 4. Compute gradient & Hessian (GPU)
        let (score, gradient, hessian) = self.compute_derivatives();

        // 5. Newton step (CPU, 6x6 solve)
        let delta = newton_step(&gradient, &hessian);

        // 6. Update transform
        transform = apply_delta(transform, delta);

        // 7. Check convergence
        if check_convergence(&delta, self.trans_epsilon, iteration, self.max_iterations) {
            break;
        }
    }

    NdtResult { transform, score, iterations, hessian }
}
```

### Tests
- [x] Convergence within 10 iterations for good initial guess
- [x] More-Thuente line search implemented and tested
- [x] Step size clamping matches Autoware behavior
- [x] Handles edge cases (no correspondences, singular Hessian)
- [ ] Final pose matches pclomp within 1cm / 0.1 degree (rosbag validation pending)

---

## Phase 4: Scoring & NVTL (1 week)

### Goal
Compute transform probability and NVTL for quality assessment.

### Components

#### 4.1 Transform Probability
```rust
#[cube(launch_unchecked)]
fn compute_transform_probability<F: Float>(
    transformed_points: &Array<F>,
    voxel_means: &Array<F>,
    voxel_inv_covs: &Array<F>,
    voxel_hash: &Array<u32>,
    gauss_d1: F,
    gauss_d2: F,
    scores: &mut Array<F>,  // Per-point scores
) {
    // Similar to derivative computation but only score
    // ...
}
```

#### 4.2 NVTL (Nearest Voxel Transformation Likelihood)
```rust
#[cube(launch_unchecked)]
fn compute_nvtl<F: Float>(
    transformed_points: &Array<F>,
    voxel_means: &Array<F>,
    voxel_inv_covs: &Array<F>,
    voxel_hash: &Array<u32>,
    gauss_d1: F,
    gauss_d2: F,
    nearest_scores: &mut Array<F>,  // Max score per point
) {
    let idx = ABSOLUTE_POS;

    let x_trans = load_vec3(transformed_points, idx);
    let mut max_score = F::new(0.0);

    for offset in NEIGHBOR_OFFSETS {
        let voxel_idx = find_voxel(x_trans, offset, voxel_hash);
        if voxel_idx >= 0 {
            let score = compute_point_score(x_trans, voxel_idx, ...);
            if score > max_score {
                max_score = score;
            }
        }
    }

    nearest_scores[idx] = max_score;
}
```

### Tests
- [x] Transform probability matches CPU implementation
- [x] Per-point scores computed correctly
- [x] NVTL neighbor search finds all relevant voxels (radius search)
- [x] NVTL vs transform probability comparison
- [x] Scoring functions match Autoware's algorithm
- [ ] GPU kernel performance: < 2ms for scoring

---

## Phase 5: Integration & Optimization (2 weeks)

### Goal
Integrate with cuda_ndt_matcher and optimize performance.

### Components

#### 5.1 API Design
```rust
// src/ndt_cuda/src/lib.rs

pub struct NdtCuda {
    client: CubeClient,
    config: NdtConfig,
    voxel_grid: Option<GpuVoxelGrid>,
}

impl NdtCuda {
    pub fn new(config: NdtConfig) -> Result<Self>;

    /// Set target (map) point cloud - builds voxel grid
    pub fn set_target(&mut self, points: &[[f32; 3]]) -> Result<()>;

    /// Align source to target with initial guess
    pub fn align(
        &self,
        source: &[[f32; 3]],
        initial_guess: Isometry3<f64>,
    ) -> Result<NdtResult>;

    /// Compute scores without optimization
    pub fn compute_score(
        &self,
        source: &[[f32; 3]],
        pose: Isometry3<f64>,
    ) -> Result<ScoreResult>;
}

pub struct NdtConfig {
    pub resolution: f32,
    pub max_iterations: u32,
    pub trans_epsilon: f64,
    pub step_size: f64,
    pub outlier_ratio: f64,
}

pub struct NdtResult {
    pub pose: Isometry3<f64>,
    pub converged: bool,
    pub score: f64,
    pub nvtl: f64,
    pub iterations: u32,
    pub hessian: Matrix6<f64>,
}
```

#### 5.2 Memory Management
- Reuse GPU buffers across calls
- Lazy voxel grid rebuild only when target changes
- Stream-ordered memory allocation

#### 5.3 Performance Optimization
- Tune CubeCL launch parameters
- Profile with Nsight Systems
- Optimize memory access patterns (coalescing)
- Consider persistent kernel approach for small workloads

### Tests
- [x] High-level NdtScanMatcher API with builder pattern
- [x] Feature flags for ndt_cuda vs fast-gicp backends
- [x] Unit tests for API (208 tests passing)
- [x] Covariance estimation with Laplace approximation
- [x] Initial pose estimation with TPE sampling
- [x] Debug output (JSONL format) for comparison
- [x] GPU runtime integration with CubeCL
- [x] `GpuRuntime` with CUDA device/client management
- [x] GPU kernel launches for transform, radius search, scoring, gradient
- [x] `use_gpu` config option with automatic fallback to CPU
- [ ] Integration test with sample rosbag (Phase 6)
- [ ] A/B comparison with pclomp (Phase 6)
- [ ] Latency benchmarking (target: < 20ms for typical workload)
- [ ] Memory usage profiling (target: < 500MB)

---

## Phase 6: Validation & Production (2 weeks)

### Goal
Validate against pclomp and prepare for production.

### Components

#### 6.1 Numerical Validation
- Compare every intermediate value with pclomp
- Log divergence points
- Create regression test suite

#### 6.2 Edge Cases
- Empty point clouds
- Single-voxel scenes
- Degenerate covariances
- Large initial pose errors

#### 6.3 Documentation
- API documentation
- Performance tuning guide
- Troubleshooting guide

---

## Phase 7: ROS Integration & Production Features

### Goal
Complete ROS integration features for full Autoware compatibility.

### 7.1 TF Broadcasting (Priority: High) ✅ COMPLETE

**Implemented:** `map → ndt_base_link` transform broadcast matching Autoware.

**Implementation Details:**
- Added `tf_pub: Publisher<TFMessage>` to `DebugPublishers` struct
- Publisher publishes to `/tf` topic (absolute topic name)
- `publish_tf()` function converts `Pose` to `TransformStamped`
- Frame IDs are configurable via `frame.map_frame` and `frame.ndt_base_frame` parameters
- Default: `map` → `ndt_base_link` (matching Autoware defaults)

**Code Location:** `src/cuda_ndt_matcher/src/main.rs`

```rust
fn publish_tf(
    tf_pub: &Publisher<TFMessage>,
    stamp: &builtin_interfaces::msg::Time,
    pose: &Pose,
    map_frame: &str,
    ndt_base_frame: &str,
) {
    let transform = geometry_msgs::msg::Transform {
        translation: geometry_msgs::msg::Vector3 {
            x: pose.position.x,
            y: pose.position.y,
            z: pose.position.z,
        },
        rotation: pose.orientation.clone(),
    };

    let transform_stamped = geometry_msgs::msg::TransformStamped {
        header: Header {
            stamp: stamp.clone(),
            frame_id: map_frame.to_string(),
        },
        child_frame_id: ndt_base_frame.to_string(),
        transform,
    };

    let tf_msg = TFMessage {
        transforms: vec![transform_stamped],
    };

    tf_pub.publish(&tf_msg);
}
```

### 7.2 Dynamic Map Loading (Priority: High) ✅ COMPLETE

**Implemented:** Full differential map loading via `GetDifferentialPointCloudMap` service.

**What's Working:**
- `MapUpdateModule` with tile management and position-based filtering
- `DynamicMapLoader` service client for `GetDifferentialPointCloudMap`
- Automatic service request during each NDT alignment (`on_points`)
- `should_update()` - checks if position moved beyond `update_distance`
- `out_of_map_range()` - detects when approaching edge of loaded map
- `check_and_update()` - combined check+update with NDT target refresh
- `get_stats()` - map statistics for monitoring
- Points filtered within `map_radius` of current position
- Async service callback handling (node spins for response processing)

**Code Locations:**
- `src/cuda_ndt_matcher/src/map_module.rs` - `MapUpdateModule` and `DynamicMapLoader`
- `src/cuda_ndt_matcher/src/main.rs` - Integration in `on_points()`

**Service Client Implementation:**
```rust
// DynamicMapLoader handles the GetDifferentialPointCloudMap service
pub struct DynamicMapLoader {
    client: Client<GetDifferentialPointCloudMap>,
    map_module: Arc<MapUpdateModule>,
    request_pending: Arc<AtomicBool>,
}

impl DynamicMapLoader {
    // Create service client for pcd_loader_service
    pub fn new(node: &Node, service_name: &str, map_module: Arc<MapUpdateModule>) -> Result<Self>;

    // Request map tiles around position (async, callback-based)
    pub fn request_map_update(&self, position: &Point, map_radius: f32) -> Result<bool>;
}
```

**Behavior:**
1. Service client connects to `pcd_loader_service`
2. On position change beyond `update_distance`, request is sent
3. Request includes current position, radius, and cached tile IDs
4. Response provides new tiles to add and old tile IDs to remove
5. Callback updates `MapUpdateModule` with differential changes
6. Node spinning ensures callback execution

**NOT Implemented:**
- Secondary NDT for non-blocking updates (Autoware feature for smoother transitions)

### 7.3 GNSS Regularization (Priority: Medium)

Autoware uses GNSS poses to regularize NDT in open areas where scan matching may drift.

```rust
// Add regularization term to NDT cost function
pub struct RegularizationTerm {
    gnss_pose: Option<Isometry3<f64>>,
    scale_factor: f64,  // Default: 0.01
}

impl RegularizationTerm {
    pub fn add_to_derivatives(
        &self,
        current_pose: &Isometry3<f64>,
        gradient: &mut Vector6<f64>,
        hessian: &mut Matrix6<f64>,
    ) {
        if let Some(gnss) = &self.gnss_pose {
            // Add quadratic penalty: scale * ||current - gnss||^2
            let diff = pose_difference(current_pose, gnss);
            *gradient += self.scale_factor * diff;
            // Hessian contribution: scale * I
        }
    }
}
```

### 7.4 Multi-NDT Covariance Estimation (Priority: Medium) ✅ COMPLETE

**Implemented:** Full multi-NDT covariance estimation matching Autoware's algorithm.

**Modes Supported:**
- `MULTI_NDT`: Run NDT alignment from offset poses, compute sample covariance
- `MULTI_NDT_SCORE`: Compute NVTL at offset poses (no alignment), use softmax-weighted covariance

**Code Location:** `src/cuda_ndt_matcher/src/covariance.rs`

**Key Functions:**
```rust
/// Create offset poses rotated by result pose orientation
pub fn propose_offset_poses(
    result_pose: &Pose,
    offset_x: &[f64],
    offset_y: &[f64],
) -> Vec<Pose>;

/// MULTI_NDT: Run alignment from each offset, compute sample covariance
pub fn estimate_xy_covariance_by_multi_ndt(
    ndt_manager: &mut NdtManager,
    sensor_points: &[[f32; 3]],
    map_points: &[[f32; 3]],
    result_pose: &Pose,
    estimation_params: &CovarianceEstimationParams,
) -> MultiNdtResult;

/// MULTI_NDT_SCORE: Compute NVTL at offsets, use softmax weights
pub fn estimate_xy_covariance_by_multi_ndt_score(
    ndt_manager: &mut NdtManager,
    sensor_points: &[[f32; 3]],
    map_points: &[[f32; 3]],
    result_pose: &Pose,
    estimation_params: &CovarianceEstimationParams,
) -> MultiNdtResult;
```

**Default Offset Model (matching Autoware):**
- X offsets: [0.0, 0.0, 0.5, -0.5, 1.0, -1.0]
- Y offsets: [0.5, -0.5, 0.0, 0.0, 0.0, 0.0]

**Integration:**
- `estimate_covariance_full()` handles all modes including multi-NDT
- Falls back to Laplace approximation if required data is not available

### 7.5 Diagnostics Interface (Priority: Low) ✅ Complete

Add ROS diagnostics for system health monitoring.

**Implementation:**
- `DiagnosticsInterface` publishes to `/diagnostics` topic
- `DiagnosticCategory` for each diagnostic type with key-value pairs and severity levels
- `ScanMatchingDiagnostics` collects scan matching metrics:
  - `sensor_points_size`, `sensor_points_delay_time_sec`, `sensor_points_max_distance`
  - `is_activated`, `is_set_map_points`, `is_succeed_interpolate_initial_pose`
  - `iteration_num`, `oscillation_count`, `transform_probability`, `nvtl`
  - `distance_initial_to_result`, `execution_time_ms`, `skipping_publish_num`
- Diagnostic levels: OK, WARN (oscillation, no map, etc.), ERROR (transform failed)

### 7.6 Oscillation Detection (Priority: Low)

Detect when optimization oscillates between poses.

```rust
pub fn count_oscillation(pose_history: &[Pose]) -> usize {
    // Count direction reversals in pose sequence
    let mut reversals = 0;
    for window in pose_history.windows(3) {
        let d1 = pose_difference(&window[0], &window[1]);
        let d2 = pose_difference(&window[1], &window[2]);
        if d1.dot(&d2) < 0.0 {
            reversals += 1;
        }
    }
    reversals
}
```

### 7.7 Monte Carlo Visualization (Priority: Low) ✅ COMPLETE

Publish visualization markers for Monte Carlo initial pose estimation particles.

**Implementation:**
- `create_monte_carlo_markers()` converts particles to `MarkerArray`
- Initial poses shown as small blue spheres
- Result poses shown as spheres colored by score (red=low, green=high)
- Best particle highlighted with larger size
- Published to `monte_carlo_initial_pose_marker` topic after NDT align service call
- Markers have 10-second lifetime for debugging visibility

### 7.8 GPU Scoring Integration (Priority: Low) ⚠️ PARTIAL

Use GPU for transform probability and NVTL evaluation.

**Status:** Transform probability uses GPU; NVTL still uses CPU.

**Completed:**
- `evaluate_transform_probability()` uses GPU when available, falls back to CPU
- `evaluate_scores_gpu()` integrated and working

**Remaining: NVTL GPU Kernel**

NVTL (Nearest Voxel Transformation Likelihood) requires a different aggregation than transform probability:
- **Transform probability:** Sum all voxel scores per point, then `total / correspondences`
- **NVTL (Autoware):** Take **max** score per point across voxels, then average

The current `compute_ndt_score_kernel` computes sum per point. To match Autoware's NVTL:

**Steps to Complete:**

1. **Add NVTL scoring kernel** (`src/ndt_cuda/src/derivatives/gpu.rs`)
   ```rust
   #[cube(launch)]
   fn compute_ndt_nvtl_kernel<F: Float>(
       // Same inputs as compute_ndt_score_kernel
       // Output: max_scores[point_idx] = max score across all neighbor voxels
   )
   ```
   - For each point, iterate through neighbors and track max score (not sum)
   - Store max score per point in output buffer

2. **Add GPU runtime method** (`src/ndt_cuda/src/runtime.rs`)
   ```rust
   pub fn compute_nvtl_scores(...) -> Result<GpuNvtlResult> {
       // Launch NVTL kernel
       // Return per-point max scores and count of points with neighbors
   }
   ```

3. **Integrate into evaluate_nvtl()** (`src/ndt_cuda/src/ndt.rs`)
   ```rust
   pub fn evaluate_nvtl(...) -> Result<f64> {
       // Try GPU path first
       if let Some(result) = self.evaluate_nvtl_gpu(source_points, pose) {
           return Ok(result);
       }
       // Fall back to CPU
       ...
   }
   ```

4. **Add tests** comparing GPU NVTL output to CPU implementation

### Tests
- [x] TF broadcast implemented (`map` → `ndt_base_link`)
- [ ] TF broadcast verified with `tf2_echo` (runtime test)
- [x] Position-based map update logic implemented
- [x] Map radius filtering works correctly
- [x] `check_and_update()` convenience method
- [x] `get_stats()` provides map statistics
- [ ] Dynamic map loading with pcd_loader service (requires autoware_map_msgs bindings)
- [x] GNSS regularization implemented (penalizes deviation from GNSS pose)
- [ ] Multi-NDT covariance matches Autoware output
- [ ] Diagnostics published to `/diagnostics`
- [x] Oscillation detection implemented (publishes to `local_optimal_solution_oscillation_num`)
- [x] Monte Carlo particle visualization (markers to `monte_carlo_initial_pose_marker`)
- [x] GPU scoring for evaluate_transform_probability
- [ ] GPU scoring for evaluate_nvtl (needs max-per-point kernel)

---

## Timeline Summary

### Completed Work

| Phase | Status | Actual Duration |
|-------|--------|-----------------|
| Phase 1: Voxel Grid | ✅ Complete | ~2 weeks |
| Phase 2: Derivatives | ✅ Complete | ~3 weeks |
| Phase 3: Newton Solver | ✅ Complete | ~1.5 weeks |
| Phase 4: Scoring | ✅ Complete | ~1 week |
| Phase 5: Integration (API) | ✅ Complete | ~1.5 weeks |

### Remaining Work

| Phase | Estimated Duration | Priority | Status |
|-------|-------------------|----------|--------|
| Phase 5: GPU Runtime | 2-3 weeks | Medium | ✅ Complete |
| Phase 6: Validation | 1-2 weeks | High | Pending |
| Phase 7.1: TF Broadcast | 2 days | High | ✅ Complete |
| Phase 7.2: Dynamic Map Loading | 1 week | High | ✅ Complete |
| Phase 7.3: GNSS Regularization | 3-4 days | Medium | ✅ Complete |
| Phase 7.4: Multi-NDT Covariance | 2-3 days | Medium | ✅ Complete |
| Phase 7.5: Diagnostics | 1 day | Low | ✅ Complete |
| Phase 7.6: Oscillation Detection | 1 day | Low | ✅ Complete |
| Phase 7.7: Monte Carlo Visualization | 0.5 day | Low | ✅ Complete |
| Phase 7.8: GPU Scoring Integration | 1 day | Low | ⚠️ Partial (TP done, NVTL pending) |
| **Total Remaining** | **~1-2 weeks** | | Phase 6 + minor gaps |

### Priority Order

1. **Phase 6: Validation** - Run rosbag comparison to verify algorithm correctness
2. ~~**Phase 7.1: TF Broadcast** - Essential for Autoware stack integration~~ ✅ Complete
3. ~~**Phase 7.2: Dynamic Map Loading** - Required for production with large maps~~ ✅ Complete
4. ~~**Phase 5: GPU Runtime** - Performance optimization~~ ✅ Complete
5. ~~**Phase 7.3: GNSS Regularization**~~ ✅ Complete
6. ~~**Phase 7.6: Oscillation Detection**~~ ✅ Complete
7. ~~**Phase 7.5: Diagnostics**~~ ✅ Complete
8. ~~**Phase 7.7: Monte Carlo Visualization** - Debug visualization (optional)~~ ✅ Complete
9. **Phase 7.8: GPU Scoring Integration** - Transform probability done, NVTL kernel pending

---

## Code TODOs

Outstanding TODO comments in the codebase that represent integration or improvement work:

| Location | Description | Priority |
|----------|-------------|----------|
| `src/ndt_cuda/src/ndt.rs` | `TODO(gpu-nvtl)`: GPU-accelerated NVTL scoring | Low |
| `src/ndt_cuda/src/derivatives/gpu.rs` | `TODO(gpu-nvtl)`: Add `compute_ndt_nvtl_kernel` for max-per-point scoring | Low |
| `src/ndt_cuda/src/runtime.rs` | `TODO(gpu-nvtl)`: Add `compute_nvtl_scores()` method | Low |
| `src/ndt_cuda/src/optimization/solver.rs:431` | Proper score normalization based on number of points | Low |
| `src/cuda_ndt_matcher/src/main.rs:990` | Map loading currently waits for pcd_loader_service; add test mode with dummy map | Low |

### Integration Tasks

These are the key integration items needed for full Autoware compatibility:

1. **GPU Scoring Path** (`ndt.rs`) ⚠️ Partial
   - ✅ `evaluate_transform_probability()` uses GPU when available
   - ❌ `evaluate_nvtl()` uses CPU - needs NVTL kernel (max-per-point, not sum)
   - See Phase 7.8 for implementation steps

2. **Score Normalization** (`solver.rs:431`)
   - Currently: Raw score returned without normalization
   - Target: Normalize by number of correspondences for consistent comparison
   - Note: May affect convergence thresholds

3. **Test Map Loading** (`main.rs:990`)
   - Currently: Node waits for map from `pcd_loader_service`
   - Target: Add option to load map from file for standalone testing
   - Benefit: Easier debugging without full Autoware stack

---

## Dependencies

### Rust Crates
```toml
[dependencies]
cubecl = { version = "0.4", features = ["cuda"] }
cubecl-cuda = "0.4"
nalgebra = "0.33"
```

### Build Requirements
- CUDA Toolkit 12.x
- Rust nightly (for CubeCL proc macros)
- NVIDIA GPU (compute capability 7.0+)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| CubeCL alpha stability | Pin version, contribute fixes upstream |
| Eigendecomposition complexity | Start with simplified regularization |
| Hash table performance | Profile early, consider alternatives |
| Numerical precision | Use f64 for accumulation, f32 for storage |

---

## Success Criteria

### Algorithm (Verified ✅)
1. **Core Algorithm**: Multi-voxel radius search matches Autoware's pclomp ✅
2. **Gaussian Parameters**: d1, d2, d3 match Magnusson 2009 exactly ✅
3. **Score/Gradient/Hessian**: Match Autoware's equations ✅
4. **Convergence**: < 10 iterations for typical scenarios ✅
5. **Stability**: Handles edge cases (singular Hessian, no correspondences) ✅

### Integration (Pending)
1. **Pose Accuracy**: Final pose within 1cm / 0.1° of pclomp (rosbag validation)
2. **ROS Interface**: TF broadcast, diagnostics, full topic compatibility
3. **Large Maps**: Dynamic tile-based loading with pcd_loader service
4. **Reliability**: No crashes during continuous operation

### Performance (Pending GPU)
1. **Latency**: < 20ms for typical workload (currently ~50ms on CPU)
2. **Memory**: < 500MB GPU memory usage
3. **Throughput**: 10+ Hz alignment rate

---

## References

1. Magnusson, M. (2009). The Three-Dimensional Normal-Distributions Transform. PhD Thesis.
2. [CubeCL Documentation](https://github.com/tracel-ai/cubecl)
3. [Autoware NDT Implementation](https://github.com/autowarefoundation/autoware.universe)
4. [More-Thuente Line Search](https://www.ii.uib.no/~lennMDL/talks/MT-paper.pdf)

---

## Sources

- [CubeCL GitHub](https://github.com/tracel-ai/cubecl)
- [Rust-CUDA Project](https://github.com/Rust-GPU/Rust-CUDA)
- [Burn Deep Learning Framework](https://burn.dev/blog/going-big-and-small-for-2025/)
- [CubeCL Architecture Overview](https://gist.github.com/nihalpasham/570d4fe01b403985e1eaf620b6613774)
