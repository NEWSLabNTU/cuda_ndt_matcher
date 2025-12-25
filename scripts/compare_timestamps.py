#!/usr/bin/env python3
"""
Compare message timestamps across ROS topics to find timestamp drift.

This script subscribes to multiple topics, records wall clock time vs message
stamp for each, and reports topics with drastic timestamp shifts.

Usage:
    ros2 run --prefix 'python3' cuda_ndt_matcher compare_timestamps.py
    # Or directly:
    python3 compare_timestamps.py
"""

import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_message


@dataclass
class TopicStats:
    """Statistics for a single topic."""
    samples: list = field(default_factory=list)  # [(wall_time, msg_stamp), ...]
    msg_type: str = ""

    def add_sample(self, wall_time: float, msg_stamp: float):
        self.samples.append((wall_time, msg_stamp))

    def get_offset_stats(self) -> tuple[float, float, float] | None:
        """Return (mean_offset, min_offset, max_offset) or None if no samples."""
        if not self.samples:
            return None
        offsets = [wall - msg for wall, msg in self.samples]
        return (
            sum(offsets) / len(offsets),
            min(offsets),
            max(offsets),
        )


class TimestampComparer(Node):
    def __init__(self):
        super().__init__('timestamp_comparer')

        self.stats: dict[str, TopicStats] = defaultdict(TopicStats)
        self.subscriptions_list = []

        # Topics to monitor (with expected message types)
        self.topics_to_check = [
            '/clock',
            '/sensing/gnss/ublox/fix_velocity',
            '/sensing/gnss/ublox/nav_sat_fix',
            '/sensing/gnss/ublox/navpvt',
            '/sensing/imu/tamagawa/imu_raw',
            '/sensing/lidar/left/velodyne_packets',
            '/sensing/lidar/right/velodyne_packets',
            '/sensing/lidar/top/velodyne_packets',
            '/vehicle/status/control_mode',
            '/vehicle/status/gear_status',
            '/vehicle/status/steering_status',
            '/vehicle/status/velocity_status',
        ]

        # QoS profile for sensor data
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # Wait for topics to be available and subscribe
        self.get_logger().info("Waiting for topics...")
        self.create_timer(1.0, self.discover_and_subscribe)
        self.create_timer(5.0, self.print_stats)

        self.start_time = time.time()

    def discover_and_subscribe(self):
        """Discover available topics and subscribe to them."""
        available_topics = dict(self.get_topic_names_and_types())

        for topic in self.topics_to_check:
            if topic in available_topics and topic not in [s.topic_name for s in self.subscriptions_list]:
                msg_types = available_topics[topic]
                if msg_types:
                    msg_type_str = msg_types[0]
                    try:
                        msg_class = get_message(msg_type_str)
                        self.create_subscription_for_topic(topic, msg_class, msg_type_str)
                    except Exception as e:
                        self.get_logger().warn(f"Could not subscribe to {topic}: {e}")

    def create_subscription_for_topic(self, topic: str, msg_class: Any, msg_type_str: str):
        """Create a subscription for a topic."""
        self.stats[topic].msg_type = msg_type_str

        def callback(msg):
            self.on_message(topic, msg)

        sub = self.create_subscription(
            msg_class,
            topic,
            callback,
            self.sensor_qos
        )
        self.subscriptions_list.append(sub)
        self.get_logger().info(f"Subscribed to {topic} [{msg_type_str}]")

    def on_message(self, topic: str, msg: Any):
        """Handle incoming message, extract timestamp."""
        wall_time = time.time()

        # Try to extract timestamp from message
        msg_stamp = self.extract_timestamp(msg)

        if msg_stamp is not None:
            self.stats[topic].add_sample(wall_time, msg_stamp)

    def extract_timestamp(self, msg: Any) -> float | None:
        """Extract timestamp from message as seconds since epoch."""
        # Try common timestamp locations

        # rosgraph_msgs/Clock
        if hasattr(msg, 'clock'):
            return self.stamp_to_seconds(msg.clock)

        # Messages with header.stamp
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            return self.stamp_to_seconds(msg.header.stamp)

        # Messages with stamp directly (e.g., velodyne_msgs)
        if hasattr(msg, 'stamp'):
            return self.stamp_to_seconds(msg.stamp)

        # sensor_msgs/NavSatFix and similar
        if hasattr(msg, 'header'):
            header = msg.header
            if hasattr(header, 'stamp'):
                return self.stamp_to_seconds(header.stamp)

        # Some messages have timestamps in data fields
        if hasattr(msg, 'time_stamp'):
            return float(msg.time_stamp)

        return None

    def stamp_to_seconds(self, stamp) -> float:
        """Convert ROS stamp to seconds since epoch."""
        if hasattr(stamp, 'sec') and hasattr(stamp, 'nanosec'):
            return float(stamp.sec) + float(stamp.nanosec) / 1e9
        elif hasattr(stamp, 'secs') and hasattr(stamp, 'nsecs'):
            return float(stamp.secs) + float(stamp.nsecs) / 1e9
        return 0.0

    def print_stats(self):
        """Print current statistics."""
        elapsed = time.time() - self.start_time

        print("\n" + "=" * 80)
        print(f"Timestamp Comparison Report (elapsed: {elapsed:.1f}s)")
        print("=" * 80)
        print(f"{'Topic':<50} {'Samples':>8} {'Offset (s)':>12} {'Range':>20}")
        print("-" * 80)

        # Collect stats and sort by offset magnitude
        topic_offsets = []
        for topic, stats in sorted(self.stats.items()):
            offset_stats = stats.get_offset_stats()
            if offset_stats:
                mean_off, min_off, max_off = offset_stats
                topic_offsets.append((topic, len(stats.samples), mean_off, min_off, max_off))

        if not topic_offsets:
            print("No samples collected yet...")
            return

        # Sort by absolute mean offset (largest first)
        topic_offsets.sort(key=lambda x: abs(x[2]), reverse=True)

        for topic, count, mean_off, min_off, max_off in topic_offsets:
            # Flag topics with large offsets (> 1 year difference suggests wrong epoch)
            flag = ""
            if abs(mean_off) > 365 * 24 * 3600:  # > 1 year
                flag = " *** WRONG EPOCH ***"
            elif abs(mean_off) > 24 * 3600:  # > 1 day
                flag = " ** LARGE OFFSET **"
            elif abs(mean_off) > 1.0:  # > 1 second
                flag = " * DRIFT *"

            range_str = f"[{min_off:+.3f}, {max_off:+.3f}]"
            print(f"{topic:<50} {count:>8} {mean_off:>+12.3f} {range_str:>20}{flag}")

        print("-" * 80)

        # Show reference time info
        if '/clock' in self.stats and self.stats['/clock'].samples:
            _, last_clock = self.stats['/clock'].samples[-1]
            print(f"Last /clock stamp: {last_clock:.3f} ({time.ctime(last_clock)})")
        print(f"Current wall time: {time.time():.3f} ({time.ctime()})")


def main():
    rclpy.init()
    node = TimestampComparer()

    try:
        print("Starting timestamp comparison...")
        print("Press Ctrl+C to stop and see final report")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.print_stats()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
