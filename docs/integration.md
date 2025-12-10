# Integration

## fast_gicp_rust

### Cargo.toml

```toml
[dependencies]
fast-gicp = { path = "../external/fast_gicp_rust/fast-gicp", features = ["cuda"] }
```

### NDT Registration

```rust
use fast_gicp::{NDTCuda, PointCloudXYZ, DistanceMode, NeighborSearchMethod};

let ndt = NDTCuda::builder()
    .resolution(1.0)
    .max_iterations(35)
    .transformation_epsilon(0.01)
    .distance_mode(DistanceMode::P2D)
    .neighbor_search_method(NeighborSearchMethod::Direct7)
    .build()?;

let result = ndt.align_with_guess(&source, &target, &initial_pose)?;
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resolution` | 1.0 | Voxel size (m) |
| `max_iterations` | 35 | Max iterations |
| `transformation_epsilon` | 0.01 | Convergence threshold |
| `distance_mode` | P2D | P2D or D2D |
| `neighbor_search_method` | Direct7 | Voxel search pattern |

## cubecl (for additional features)

### Cargo.toml

```toml
[dependencies]
cubecl = { version = "0.2", features = ["cuda"] }
cubecl-cuda = "0.2"
```

### Kernel Example (Multi-NDT Covariance)

```rust
#[cube(launch_unchecked)]
fn compute_ndt_scores<F: Float>(
    particle_poses: &Array<F>,    // [P × 6] initial poses
    scores: &mut Array<F>,        // [P] output scores
    // ... voxel grid data
) {
    let particle_idx = ABSOLUTE_POS;
    // Compute NDT score for this particle's pose
}
```

### Use Cases for cubecl

1. **Multi-NDT covariance estimation**
   - Launch NDT from multiple perturbed poses
   - Collect convergence points
   - Compute 2D covariance from distribution

2. **Monte Carlo initial pose**
   - Generate particle cloud around initial guess
   - Evaluate likelihood for each particle
   - Select best candidates

3. **No-ground score computation**
   - Filter ground points on GPU
   - Compute separate alignment score

## Build Requirements

```bash
# System dependencies
sudo apt install libpcl-dev libeigen3-dev

# CUDA 11.0+ required for fast_gicp cuda feature
nvcc --version

# Build
just build
```
