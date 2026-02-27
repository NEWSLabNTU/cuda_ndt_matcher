//! Shared pose/quaternion conversion utilities.
//!
//! Eliminates duplicate geometry_msgs <-> nalgebra conversions
//! scattered across main.rs, covariance.rs, and initial_pose.rs.

use geometry_msgs::msg::{Point, Pose};
use nalgebra::{Isometry3, Quaternion as NaQuaternion, Translation3, UnitQuaternion};

/// Convert a geometry_msgs Quaternion to a nalgebra UnitQuaternion.
pub(crate) fn unit_quat_from_msg(q: &geometry_msgs::msg::Quaternion) -> UnitQuaternion<f64> {
    UnitQuaternion::from_quaternion(NaQuaternion::new(q.w, q.x, q.y, q.z))
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
    let q = iso.rotation.quaternion();
    Pose {
        position: Point {
            x: t.x,
            y: t.y,
            z: t.z,
        },
        orientation: geometry_msgs::msg::Quaternion {
            x: q.i,
            y: q.j,
            z: q.k,
            w: q.w,
        },
    }
}

/// Extract Euler angles (roll, pitch, yaw) in radians from a geometry_msgs Pose.
#[cfg(feature = "debug-output")]
pub(crate) fn euler_from_pose(pose: &Pose) -> (f64, f64, f64) {
    unit_quat_from_msg(&pose.orientation).euler_angles()
}
