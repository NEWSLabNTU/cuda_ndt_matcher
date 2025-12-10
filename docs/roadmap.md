# Roadmap

## Phase 1: Core Integration

Build minimal working node using fast_gicp_rust.

### Work Items

- [ ] Add fast-gicp dependency to workspace
- [ ] Implement point cloud conversion (PointCloud2 <-> fast_gicp)
- [ ] Implement basic NDT alignment using NDTCuda
- [ ] Implement pose output (ndt_pose topic)
- [ ] Add basic parameters (resolution, max_iterations)

### Passing Criteria

- Node subscribes to `points_raw` and `ekf_pose_with_covariance`
- Node publishes `ndt_pose` with valid transform
- Alignment runs on GPU (nvidia-smi shows usage)

### Tests

```bash
# Unit: point cloud conversion
cargo test point_cloud_conversion

# Integration: node starts and publishes
ros2 run cuda_ndt_matcher cuda_ndt_matcher &
ros2 topic hz /ndt_pose  # Should show messages

# Smoke: alignment with sample data
ros2 bag play sample.bag
# Verify ndt_pose output is reasonable
```

---

## Phase 2: Full ROS Interface

Match Autoware ndt_scan_matcher interface.

### Work Items

- [ ] Add all subscriptions (regularization_pose)
- [ ] Add ndt_pose_with_covariance publisher
- [ ] Implement trigger_node_srv service
- [ ] Add pcd_loader_service client
- [ ] Implement all parameters from config
- [ ] Add launch file with remapping

### Passing Criteria

- Launch file is drop-in replacement for Autoware
- All topics/services match original names
- Parameters loaded from YAML

### Tests

```bash
# Integration: launch file works
ros2 launch cuda_ndt_matcher ndt_scan_matcher.launch.xml

# Interface: topics exist
ros2 topic list | grep -E "ndt_pose|points_raw"
ros2 service list | grep trigger_node

# Config: parameters loaded
ros2 param list /ndt_scan_matcher
```

---

## Phase 3: Covariance Estimation

Implement GPU covariance estimation using cubecl.

### Work Items

- [ ] Add cubecl dependency
- [ ] Implement FIXED covariance mode
- [ ] Implement LAPLACE approximation kernel
- [ ] Implement MULTI_NDT kernel
- [ ] Implement MULTI_NDT_SCORE kernel
- [ ] Add covariance parameters

### Passing Criteria

- All 4 covariance modes work
- Output covariance varies with alignment quality
- GPU kernel execution confirmed

### Tests

```bash
# Unit: covariance kernels
cargo test covariance --features cuda

# Integration: covariance output
ros2 param set /ndt_scan_matcher covariance.covariance_estimation_type 1
ros2 topic echo /ndt_pose_with_covariance --field pose.covariance

# Benchmark: kernel performance
cargo bench covariance
```

---

## Phase 4: Initial Pose Estimation

Implement Monte Carlo initial pose using cubecl.

### Work Items

- [ ] Implement particle generation kernel
- [ ] Implement score evaluation kernel
- [ ] Implement best particle selection
- [ ] Add TPE (Tree-Structured Parzen Estimator) search
- [ ] Add initial_pose_estimation parameters

### Passing Criteria

- Node recovers from poor initial guess
- particles_num affects search coverage
- GPU parallel evaluation confirmed

### Tests

```bash
# Unit: particle sampling
cargo test initial_pose --features cuda

# Integration: recovery from bad initial
ros2 topic pub /ekf_pose_with_covariance ... --once  # Bad pose
# Verify ndt_pose converges to correct location

# Benchmark: particle evaluation
cargo bench initial_pose
```

---

## Phase 5: Dynamic Map Loading

Implement map management and caching.

### Work Items

- [ ] Implement map cache structure
- [ ] Implement distance-based update trigger
- [ ] Add pcd_loader_service client
- [ ] Implement map radius filtering
- [ ] Add dynamic_map_loading parameters

### Passing Criteria

- Map updates when vehicle moves update_distance
- Only loads map within map_radius
- Old map sections unloaded

### Tests

```bash
# Integration: map loading triggers
ros2 service call /pcd_loader_service ...
# Verify map updates in node

# Memory: map cache bounded
# Monitor memory usage during long run
```

---

## Phase 6: Validation & Diagnostics

Implement all validation and diagnostic features.

### Work Items

- [ ] Implement timestamp validation
- [ ] Implement distance validation
- [ ] Implement execution time monitoring
- [ ] Implement skip counting
- [ ] Add diagnostics publisher
- [ ] Add no_ground_points score option

### Passing Criteria

- Invalid inputs rejected with diagnostics
- Execution time warnings published
- No ground score computed when enabled

### Tests

```bash
# Unit: validation logic
cargo test validation

# Integration: diagnostics output
ros2 topic echo /diagnostics

# Edge cases: invalid inputs handled
ros2 topic pub /points_raw ...  # Empty cloud
ros2 topic pub /ekf_pose_with_covariance ...  # Old timestamp
```

---

## Phase 7: Performance Optimization

Optimize for real-time performance.

### Work Items

- [ ] Profile GPU kernel execution
- [ ] Optimize memory transfers
- [ ] Implement async processing
- [ ] Add performance metrics
- [ ] Upstream improvements to fast_gicp_rust

### Passing Criteria

- < 50ms total latency for 10K points
- < 100ms for 50K points
- Stable frame rate without drops

### Tests

```bash
# Benchmark: end-to-end latency
cargo bench ndt_pipeline

# Stress: high frequency input
ros2 bag play --rate 2.0 sample.bag
# Verify no message drops

# Profile: GPU utilization
nsys profile ros2 run cuda_ndt_matcher cuda_ndt_matcher
```
