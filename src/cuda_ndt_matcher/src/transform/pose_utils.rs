//! Shared pose/quaternion conversion utilities.
//!
//! Eliminates duplicate geometry_msgs <-> nalgebra conversions
//! scattered across main.rs, covariance.rs, and initial_pose.rs.

use geometry_msgs::msg::{Point, Pose, PoseWithCovarianceStamped};
use nalgebra::{Isometry3, Quaternion as NaQuaternion, Translation3, UnitQuaternion};
use rayon::prelude::*;

/// Convert a ROS timestamp to nanoseconds (i64).
pub(crate) fn stamp_to_ns(stamp: &builtin_interfaces::msg::Time) -> i64 {
    stamp.sec as i64 * 1_000_000_000 + stamp.nanosec as i64
}

/// Convert a ROS timestamp to nanoseconds (u64).
pub(crate) fn stamp_to_ns_u64(stamp: &builtin_interfaces::msg::Time) -> u64 {
    stamp.sec as u64 * 1_000_000_000 + stamp.nanosec as u64
}

/// Convert a geometry_msgs Quaternion to a nalgebra UnitQuaternion.
pub(crate) fn unit_quat_from_msg(q: &geometry_msgs::msg::Quaternion) -> UnitQuaternion<f64> {
    UnitQuaternion::from_quaternion(NaQuaternion::new(q.w, q.x, q.y, q.z))
}

/// Convert a nalgebra UnitQuaternion to a geometry_msgs Quaternion.
pub(crate) fn quat_to_msg(q: &UnitQuaternion<f64>) -> geometry_msgs::msg::Quaternion {
    geometry_msgs::msg::Quaternion {
        x: q.i,
        y: q.j,
        z: q.k,
        w: q.w,
    }
}

/// Euclidean distance between two geometry_msgs Points.
pub(crate) fn point_distance(a: &Point, b: &Point) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Extract position Point from a PoseWithCovarianceStamped.
pub(crate) fn position_from_pose_cov(p: &PoseWithCovarianceStamped) -> Point {
    Point {
        x: p.pose.pose.position.x,
        y: p.pose.pose.position.y,
        z: p.pose.pose.position.z,
    }
}

/// Bulk transform `[f32; 3]` points by an `Isometry3<f64>`.
///
/// Uses rayon parallel iteration for large point clouds (>4096 points)
/// to avoid thread-spawning overhead on small inputs.
pub(crate) fn transform_points_f32(points: &[[f32; 3]], tf: &Isometry3<f64>) -> Vec<[f32; 3]> {
    let transform_fn = |p: &[f32; 3]| {
        let pt = nalgebra::Point3::new(p[0] as f64, p[1] as f64, p[2] as f64);
        let transformed = tf * pt;
        [
            transformed.x as f32,
            transformed.y as f32,
            transformed.z as f32,
        ]
    };

    if points.len() > 4096 {
        points.par_iter().map(transform_fn).collect()
    } else {
        points.iter().map(transform_fn).collect()
    }
}

/// Convert a geometry_msgs Pose to a nalgebra Isometry3.
pub(crate) fn isometry_from_pose(pose: &Pose) -> Isometry3<f64> {
    let p = &pose.position;
    let translation = Translation3::new(p.x, p.y, p.z);
    let rotation = unit_quat_from_msg(&pose.orientation);
    Isometry3::from_parts(translation, rotation)
}

/// Convert a nalgebra Isometry3 to a geometry_msgs Pose.
pub(crate) fn pose_from_isometry(iso: &Isometry3<f64>) -> Pose {
    let t = iso.translation;
    Pose {
        position: Point {
            x: t.x,
            y: t.y,
            z: t.z,
        },
        orientation: quat_to_msg(&iso.rotation),
    }
}

/// Extract Euler angles (roll, pitch, yaw) in radians from a geometry_msgs Pose.
#[cfg(feature = "debug-output")]
pub(crate) fn euler_from_pose(pose: &Pose) -> (f64, f64, f64) {
    unit_quat_from_msg(&pose.orientation).euler_angles()
}
