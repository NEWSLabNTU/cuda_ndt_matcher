# Rosbag Replay Simulation Guide

This guide explains how to run Autoware's rosbag replay simulation with your own rosbag data. It covers analyzing your rosbag, modifying Autoware's sensor configuration, and troubleshooting common issues.

## Prerequisites

- Autoware installed and built (see [Autoware Documentation](https://autowarefoundation.github.io/autoware-documentation/))
- A rosbag file with sensor data (LiDAR, IMU, GNSS, vehicle status)
- A point cloud map (.pcd) and lanelet2 map (.osm) for the recorded area

## Quick Start (If Your Rosbag Matches sample_sensor_kit)

If your rosbag has the same sensor configuration as Autoware's sample rosbag (4 LiDARs, Tamagawa IMU, u-blox GNSS):

```bash
# Terminal 1: Launch Autoware
source ~/autoware/install/setup.bash
ros2 launch autoware_launch logging_simulator.launch.xml \
  map_path:=/path/to/your/map \
  vehicle_model:=sample_vehicle \
  sensor_model:=sample_sensor_kit

# Terminal 2: Play rosbag
source ~/autoware/install/setup.bash
ros2 bag play /path/to/your/rosbag -r 0.5 -s sqlite3
```

If you encounter errors, your sensor configuration likely differs. Follow the steps below.

## Step 1: Analyze Your Rosbag

First, identify what sensors are recorded in your rosbag:

```bash
ros2 bag info /path/to/your/rosbag -s sqlite3
```

Look for these topic patterns:

| Sensor Type       | Expected Topic Pattern                   | Message Type                               |
|-------------------|------------------------------------------|--------------------------------------------|
| LiDAR             | `/sensing/lidar/{name}/velodyne_packets` | `velodyne_msgs/msg/VelodyneScan`           |
| LiDAR (processed) | `/sensing/lidar/{name}/pointcloud_raw`   | `sensor_msgs/msg/PointCloud2`              |
| IMU               | `/sensing/imu/{name}/imu_raw`            | `sensor_msgs/msg/Imu`                      |
| GNSS              | `/sensing/gnss/{name}/nav_sat_fix`       | `sensor_msgs/msg/NavSatFix`                |
| Vehicle velocity  | `/vehicle/status/velocity_status`        | `autoware_vehicle_msgs/msg/VelocityReport` |

**Example output:**
```
Topic: /sensing/lidar/top/velodyne_packets | Type: velodyne_msgs/msg/VelodyneScan | Count: 288
Topic: /sensing/lidar/left/velodyne_packets | Type: velodyne_msgs/msg/VelodyneScan | Count: 299
Topic: /sensing/lidar/right/velodyne_packets | Type: velodyne_msgs/msg/VelodyneScan | Count: 299
Topic: /sensing/imu/tamagawa/imu_raw | Type: sensor_msgs/msg/Imu | Count: 853
Topic: /sensing/gnss/ublox/nav_sat_fix | Type: sensor_msgs/msg/NavSatFix | Count: 30
```

This example has 3 LiDARs (top, left, right) but sample_sensor_kit expects 4 (top, left, right, **rear**).

## Step 2: Modify LiDAR Configuration

Edit `src/sensor_kit/sample_sensor_kit_launch/sample_sensor_kit_launch/launch/lidar.launch.xml`

### Remove Missing LiDARs

Delete the `<group>` block for any LiDAR not in your rosbag. For example, to remove `rear`:

```xml
<!-- DELETE THIS ENTIRE BLOCK if rear LiDAR is not in rosbag -->
<group>
  <push-ros-namespace namespace="rear"/>
  <include file="$(find-pkg-share common_sensor_launch)/launch/velodyne_VLP16.launch.xml">
    <arg name="max_range" value="1.5"/>
    <arg name="sensor_frame" value="velodyne_rear"/>
    ...
  </include>
</group>
```

### Add Additional LiDARs

If your rosbag has LiDARs not in the default config, add a new `<group>` block:

```xml
<group>
  <push-ros-namespace namespace="your_lidar_name"/>
  <include file="$(find-pkg-share common_sensor_launch)/launch/velodyne_VLP16.launch.xml">
    <arg name="max_range" value="100.0"/>
    <arg name="sensor_frame" value="velodyne_your_lidar_name"/>
    <arg name="sensor_ip" value="192.168.1.205"/>
    <arg name="host_ip" value="$(var host_ip)"/>
    <arg name="data_port" value="2372"/>
    <arg name="scan_phase" value="180.0"/>
    <arg name="launch_driver" value="$(var launch_driver)"/>
    <arg name="vehicle_mirror_param_file" value="$(var vehicle_mirror_param_file)"/>
    <arg name="container_name" value="pointcloud_container"/>
  </include>
</group>
```

## Step 3: Update Pointcloud Concatenation

Edit `src/sensor_kit/sample_sensor_kit_launch/sample_sensor_kit_launch/launch/pointcloud_preprocessor.launch.py`

Update the `input_topics` list to match your LiDARs:

```python
"input_topics": [
    "/sensing/lidar/top/pointcloud_before_sync",
    "/sensing/lidar/left/pointcloud_before_sync",
    "/sensing/lidar/right/pointcloud_before_sync",
    # Add or remove topics to match your rosbag
],
```

**Important:** The topic names here use `pointcloud_before_sync`, not `velodyne_packets`. The velodyne driver converts packets to pointclouds automatically.

## Step 4: Update Sensor Calibration (Optional)

If your sensor positions differ from the defaults, edit the calibration files:

### sensor_kit_calibration.yaml

Location: `src/sensor_kit/sample_sensor_kit_launch/sample_sensor_kit_description/config/sensor_kit_calibration.yaml`

This defines sensor positions relative to `sensor_kit_base_link`:

```yaml
sensor_kit_base_link:
  velodyne_top_base_link:
    x: 0.0
    y: 0.0
    z: 0.5
    roll: 0.0
    pitch: 0.0
    yaw: 0.0
  velodyne_left_base_link:
    x: 0.0
    y: 0.5
    z: 0.3
    roll: 0.0
    pitch: 0.0
    yaw: 1.57  # 90 degrees
  # Add entries for your sensors...
```

### sensors_calibration.yaml

Location: `src/sensor_kit/sample_sensor_kit_launch/sample_sensor_kit_description/config/sensors_calibration.yaml`

This defines the sensor_kit position relative to `base_link`:

```yaml
base_link:
  sensor_kit_base_link:
    x: 0.9
    y: 0.0
    z: 2.0
    roll: 0.0
    pitch: 0.0
    yaw: 0.0
```

## Step 5: Create individual_params Entry

Autoware's localization stack looks for calibration files in the `individual_params` package. You need to create an entry for your sensor model.

```bash
# Navigate to individual_params
cd ~/autoware/src/param/autoware_individual_params/individual_params

# Create directory for your sensor kit (use same name as sensor_model arg)
mkdir -p config/default/sample_sensor_kit  # or your custom name

# Copy calibration files
cp ~/autoware/src/sensor_kit/sample_sensor_kit_launch/sample_sensor_kit_description/config/sensors_calibration.yaml \
   config/default/sample_sensor_kit/

cp ~/autoware/src/sensor_kit/sample_sensor_kit_launch/sample_sensor_kit_description/config/sensor_kit_calibration.yaml \
   config/default/sample_sensor_kit/
```

If you're using a custom sensor model name, also update `CMakeLists.txt` to install your config directory.

## Step 6: Rebuild Autoware

After making changes, rebuild the affected packages:

```bash
cd ~/autoware
colcon build --symlink-install --packages-select \
  sample_sensor_kit_launch \
  sample_sensor_kit_description \
  individual_params \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

## Step 7: Handle Timestamp Issues

### Check for Timestamp Mismatch

Some rosbags have inconsistent timestamps where the `/clock` topic doesn't match message header timestamps. To check:

```bash
# Play rosbag briefly
ros2 bag play /path/to/rosbag -s sqlite3 &
sleep 2

# Check /clock timestamp
ros2 topic echo /clock --once

# Check a sensor message timestamp
ros2 topic echo /sensing/imu/tamagawa/imu_raw --once | grep -A2 "stamp:"

# Stop playback
pkill -f "ros2 bag play"
```

If the timestamps differ significantly (different year/month), you have a timestamp mismatch.

### Fix: Clock Republisher

Create a script that publishes `/clock` from sensor message timestamps:

```python
#!/usr/bin/env python3
"""clock_from_sensor.py - Republish /clock from sensor timestamps"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu

class ClockFromSensor(Node):
    def __init__(self):
        super().__init__('clock_from_sensor')

        clock_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        self.clock_pub = self.create_publisher(Clock, '/clock', clock_qos)

        # Subscribe to a high-frequency sensor topic
        self.create_subscription(
            Imu,
            '/sensing/imu/tamagawa/imu_raw',  # Adjust to your IMU topic
            self.callback,
            10
        )
        self.last_stamp = None
        self.get_logger().info('Clock republisher started')

    def callback(self, msg):
        stamp = msg.header.stamp
        if self.last_stamp is None or \
           stamp.sec != self.last_stamp.sec or \
           stamp.nanosec != self.last_stamp.nanosec:
            self.last_stamp = stamp
            clock_msg = Clock()
            clock_msg.clock = stamp
            self.clock_pub.publish(clock_msg)

def main():
    rclpy.init()
    node = ClockFromSensor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Then play the rosbag excluding the original `/clock`:

```bash
# Get all topics except /clock
TOPICS=$(ros2 bag info /path/to/rosbag -s sqlite3 | grep -oP '(?<=Topic: )\S+' | grep -v '^/clock$' | tr '\n' ' ')

# Terminal 1: Start clock republisher
python3 clock_from_sensor.py &

# Terminal 2: Play rosbag with specific topics
ros2 bag play /path/to/rosbag -s sqlite3 --topics $TOPICS
```

## Step 8: Run the Simulation

### Terminal 1: Launch Autoware

```bash
source ~/autoware/install/setup.bash
ros2 launch autoware_launch logging_simulator.launch.xml \
  map_path:=/path/to/your/map \
  vehicle_model:=sample_vehicle \
  sensor_model:=sample_sensor_kit \
  rviz:=true
```

Wait for the map to load (you'll see the point cloud map in RViz).

### Terminal 2: Play Rosbag

```bash
source ~/autoware/install/setup.bash
ros2 bag play /path/to/rosbag -r 0.5 -s sqlite3
```

The `-r 0.5` plays at half speed. Adjust as needed.

### Terminal 3: Initialize Pose (if needed)

If localization doesn't start automatically, you may need to set an initial pose:

**Option A: Use RViz**
1. Click "2D Pose Estimate" in RViz toolbar
2. Click and drag on the map to set initial position and orientation

**Option B: Use service call**
```bash
ros2 service call /localization/pose_estimator/trigger_node std_srvs/srv/SetBool "{data: true}"
```

### Monitor Output

```bash
# Check NDT pose output
ros2 topic echo /localization/pose_estimator/pose_with_covariance

# Check topic frequencies
ros2 topic hz /localization/pose_estimator/pose_with_covariance
```

## Troubleshooting

### Error: "Failed to get transform from base_link to velodyne_X"

**Cause:** LiDAR defined in launch but not in rosbag, or missing TF.

**Solution:**
1. Remove the LiDAR from `lidar.launch.xml`
2. Ensure `vehicle:=true` in launch (publishes TF tree)

### Error: "Could not find sensors_calibration.yaml"

**Cause:** Missing `individual_params` entry for your sensor model.

**Solution:** Create the calibration files in `individual_params/config/default/{sensor_model}/`

### Error: "base_link frame does not exist"

**Cause:** Vehicle model not launching robot_state_publisher.

**Solution:** Ensure `vehicle:=true` in the launch command (this is the default).

### Error: "Mismatch between pose timestamp and current timestamp"

**Cause:** Rosbag `/clock` doesn't match sensor message timestamps.

**Solution:** Use the clock republisher script (Step 7).

### Pointcloud not visible in RViz

**Possible causes:**
1. Wrong Fixed Frame - Set to `base_link` or `map`
2. LiDAR topics not matching - Check `input_topics` in `pointcloud_preprocessor.launch.py`
3. Pointcloud display not added - Add PointCloud2 display for `/sensing/lidar/concatenated/pointcloud`

### Vehicle not moving on map

**Possible causes:**
1. Localization not initialized - Use "2D Pose Estimate" in RViz
2. NDT not enabled - Call trigger_node service
3. EKF not converging - Check `/localization/pose_twist_fusion_filter/` topics

## Reference: sample_sensor_kit Configuration

The default `sample_sensor_kit` expects these sensors:

| Sensor               | Topic                                   | Frame               |
|----------------------|-----------------------------------------|---------------------|
| Top LiDAR (VLS-128)  | `/sensing/lidar/top/velodyne_packets`   | `velodyne_top`      |
| Left LiDAR (VLP-16)  | `/sensing/lidar/left/velodyne_packets`  | `velodyne_left`     |
| Right LiDAR (VLP-16) | `/sensing/lidar/right/velodyne_packets` | `velodyne_right`    |
| Rear LiDAR (VLP-16)  | `/sensing/lidar/rear/velodyne_packets`  | `velodyne_rear`     |
| IMU                  | `/sensing/imu/tamagawa/imu_raw`         | `tamagawa/imu_link` |
| GNSS                 | `/sensing/gnss/ublox/nav_sat_fix`       | `gnss_link`         |
| Vehicle Status       | `/vehicle/status/velocity_status`       | -                   |

## Reference: File Locations

| Purpose             | File Path                                                                                                   |
|---------------------|-------------------------------------------------------------------------------------------------------------|
| LiDAR launch        | `src/sensor_kit/sample_sensor_kit_launch/sample_sensor_kit_launch/launch/lidar.launch.xml`                  |
| Pointcloud concat   | `src/sensor_kit/sample_sensor_kit_launch/sample_sensor_kit_launch/launch/pointcloud_preprocessor.launch.py` |
| Sensing launch      | `src/sensor_kit/sample_sensor_kit_launch/sample_sensor_kit_launch/launch/sensing.launch.xml`                |
| Sensor positions    | `src/sensor_kit/sample_sensor_kit_launch/sample_sensor_kit_description/config/sensor_kit_calibration.yaml`  |
| Sensor kit position | `src/sensor_kit/sample_sensor_kit_launch/sample_sensor_kit_description/config/sensors_calibration.yaml`     |
| Individual params   | `src/param/autoware_individual_params/individual_params/config/default/{sensor_model}/`                     |
| Logging simulator   | `src/launcher/autoware_launch/autoware_launch/launch/logging_simulator.launch.xml`                          |
