# CUDA NDT Matcher Launch

Launch files for testing CUDA NDT scan matcher with Autoware compatibility.

## Launch Files

| File | Description |
|------|-------------|
| `ndt_scan_matcher.launch.xml` | Launch CUDA NDT matcher with configurable topics |
| `ndt_scan_matcher_switcher.launch.xml` | Switch between CUDA and Autoware implementations |
| `test_with_rosbag.launch.xml` | Minimal setup for rosbag replay testing |

## Quick Start

### 1. Build the Package

```bash
source /opt/ros/humble/setup.bash
cd /path/to/cuda_ndt_matcher
just build
source install/setup.bash
```

### 2. Get Sample Data

Download Autoware sample data:

```bash
# Sample map and rosbag from Autoware
# See: https://autowarefoundation.github.io/autoware-documentation/main/tutorials/ad-hoc-simulation/rosbag-replay-simulation/

# Or use your own:
# - PCD map file (point cloud map)
# - Rosbag with /points_raw (LiDAR) and /initialpose or /ekf_pose_with_covariance
```

### 3. Test with Rosbag

**Terminal 1: Launch NDT**
```bash
ros2 launch cuda_ndt_matcher_launch test_with_rosbag.launch.xml \
  use_cuda:=true \
  map_path:=/path/to/map.pcd
```

**Terminal 2: Play Rosbag**
```bash
ros2 bag play /path/to/rosbag --clock
```

**Terminal 3: Set Initial Pose (via rviz2 or command)**
```bash
# Using rviz2: Click "2D Pose Estimate" and set initial position

# Or via command:
ros2 topic pub /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{
  header: {frame_id: 'map'},
  pose: {
    pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}},
    covariance: [0.25, 0, 0, 0, 0, 0,
                 0, 0.25, 0, 0, 0, 0,
                 0, 0, 0.0025, 0, 0, 0,
                 0, 0, 0, 0.0007, 0, 0,
                 0, 0, 0, 0, 0.0007, 0,
                 0, 0, 0, 0, 0, 0.07]
  }
}" --once
```

**Terminal 4: Enable NDT**
```bash
ros2 service call /ndt_scan_matcher/trigger_node_srv std_srvs/srv/SetBool "{data: true}"
```

**Terminal 5: Monitor Output**
```bash
ros2 topic echo /ndt_pose
```

## Switching Implementations

Compare CUDA vs Autoware original:

```bash
# CUDA implementation
ros2 launch cuda_ndt_matcher_launch ndt_scan_matcher_switcher.launch.xml use_cuda:=true

# Autoware original
ros2 launch cuda_ndt_matcher_launch ndt_scan_matcher_switcher.launch.xml use_cuda:=false
```

## Topics

### Input Topics

| Topic | Type | Description |
|-------|------|-------------|
| `points_raw` | `sensor_msgs/PointCloud2` | LiDAR point cloud |
| `ekf_pose_with_covariance` | `geometry_msgs/PoseWithCovarianceStamped` | Initial pose estimate |
| `regularization_pose_with_covariance` | `geometry_msgs/PoseWithCovarianceStamped` | GNSS pose (optional) |
| `pointcloud_map` | `sensor_msgs/PointCloud2` | Point cloud map |

### Output Topics

| Topic | Type | Description |
|-------|------|-------------|
| `ndt_pose` | `geometry_msgs/PoseStamped` | Estimated pose |
| `ndt_pose_with_covariance` | `geometry_msgs/PoseWithCovarianceStamped` | Estimated pose with covariance |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `trigger_node_srv` | `std_srvs/SetBool` | Enable/disable NDT matching |
| `ndt_align_srv` | `std_srvs/Trigger` | Trigger initial pose estimation |
| `map_update_srv` | `std_srvs/Trigger` | Trigger map update |

## Configuration

Edit `config/ndt_scan_matcher.param.yaml`:

```yaml
ndt:
  resolution: 2.0        # Voxel resolution (meters)
  max_iterations: 30     # Max optimization iterations
  trans_epsilon: 0.01    # Convergence threshold

sensor_points:
  required_distance: 10.0  # Min max distance in cloud

initial_pose_estimation:
  particles_num: 200       # Monte Carlo particles
  n_startup_trials: 100    # Random trials before TPE

dynamic_map_loading:
  update_distance: 20.0    # Trigger distance (meters)
  map_radius: 150.0        # Map loading radius (meters)
```

## Autoware Integration

For full Autoware stack integration, use remapped topics:

```bash
ros2 launch cuda_ndt_matcher_launch ndt_scan_matcher_switcher.launch.xml \
  use_cuda:=true \
  input_pointcloud:=/localization/util/downsample/pointcloud \
  input_initial_pose_topic:=/localization/pose_twist_fusion_filter/biased_pose_with_covariance \
  output_pose_with_covariance_topic:=/localization/pose_estimator/pose_with_covariance
```

## Troubleshooting

### No output from NDT
1. Check map is loaded: `ros2 topic echo /pointcloud_map --once`
2. Check initial pose received: `ros2 topic echo /ekf_pose_with_covariance --once`
3. Enable node: `ros2 service call .../trigger_node_srv std_srvs/srv/SetBool "{data: true}"`

### NDT not converging
1. Increase `max_iterations` in config
2. Check initial pose is close to true position
3. Verify point cloud has sufficient points (`required_distance`)

### Performance issues
1. Reduce `particles_num` for faster initial pose estimation
2. Increase `resolution` for coarser NDT grid
3. Use downsampled input pointcloud
