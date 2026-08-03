//! Type definitions for NDT optimization.

use crate::derivatives::DistanceMetric;
use nalgebra::{Isometry3, Matrix6, UnitQuaternion, Vector3, Vector6};

/// Configuration for NDT scan matching.
#[derive(Debug, Clone)]
pub struct NdtConfig {
    /// Voxel resolution in meters (typically 1.0 - 4.0m).
    pub resolution: f64,

    /// Maximum number of iterations.
    pub max_iterations: usize,

    /// Convergence threshold for pose delta norm.
    /// Iteration stops when ||Δp|| < trans_epsilon.
    pub trans_epsilon: f64,

    /// Maximum step length for Newton update (Autoware default: 0.1).
    ///
    /// The Newton step direction is normalized, then scaled by
    /// `min(newton_step_norm, step_size)`. This prevents large steps
    /// when far from the optimum while allowing full steps when close.
    ///
    /// NOTE: This is NOT a damping factor - it's the maximum allowed step length.
    pub step_size: f64,

    /// Probability that a point is an outlier (typically 0.55).
    pub outlier_ratio: f64,

    /// Whether to use line search for step size.
    pub use_line_search: bool,

    /// Regularization factor for Hessian when near-singular.
    pub regularization: f64,

    /// Distance metric for NDT cost function.
    pub distance_metric: DistanceMetric,
}

impl Default for NdtConfig {
    fn default() -> Self {
        Self {
            resolution: 2.0,
            max_iterations: 30,
            trans_epsilon: 0.01,
            step_size: 0.1, // Autoware default: max step length (NOT a damping factor)
            outlier_ratio: 0.55,
            use_line_search: false, // Disabled by default (matches Autoware - line search causes local minima)
            regularization: 0.0,    // Autoware uses plain SVD without regularization
            distance_metric: DistanceMetric::PointToDistribution,
        }
    }
}

impl NdtConfig {
    /// Create a new configuration with custom resolution.
    pub fn with_resolution(resolution: f64) -> Self {
        Self {
            resolution,
            ..Default::default()
        }
    }
}

/// Status of NDT optimization convergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceStatus {
    /// Converged: delta norm below threshold.
    Converged,

    /// Reached maximum iterations without convergence.
    MaxIterations,

    /// No valid correspondences found.
    NoCorrespondences,

    /// Hessian is singular (cannot compute Newton step).
    SingularHessian,

    /// Score increased (diverging).
    Diverged,
}

impl ConvergenceStatus {
    /// Check if the optimization converged successfully.
    pub fn is_converged(&self) -> bool {
        matches!(self, ConvergenceStatus::Converged)
    }

    /// Check if the result is usable (converged or max iterations).
    pub fn is_usable(&self) -> bool {
        matches!(
            self,
            ConvergenceStatus::Converged | ConvergenceStatus::MaxIterations
        )
    }
}

/// Result of NDT scan matching.
#[derive(Debug, Clone)]
pub struct NdtResult {
    /// Final pose estimate (transformation from source to target frame).
    pub pose: Isometry3<f64>,

    /// Convergence status.
    pub status: ConvergenceStatus,

    /// Final NDT score (higher is better, maximum at perfect alignment).
    pub score: f64,

    /// Transform probability (normalized score).
    pub transform_probability: f64,

    /// Nearest voxel transformation likelihood (NVTL).
    pub nvtl: f64,

    /// Number of iterations performed.
    pub iterations: usize,

    /// Final Hessian matrix (useful for covariance estimation).
    pub hessian: Matrix6<f64>,

    /// Number of valid correspondences in final iteration.
    pub num_correspondences: usize,

    /// Maximum consecutive oscillation count detected during optimization.
    /// Oscillation indicates the optimizer is bouncing between poses,
    /// potentially stuck in a local minimum.
    pub oscillation_count: usize,
}

impl NdtResult {
    /// Create a result indicating no correspondences were found.
    pub fn no_correspondences(initial_pose: Isometry3<f64>) -> Self {
        Self {
            pose: initial_pose,
            status: ConvergenceStatus::NoCorrespondences,
            score: 0.0,
            transform_probability: 0.0,
            nvtl: 0.0,
            iterations: 0,
            hessian: Matrix6::zeros(),
            num_correspondences: 0,
            oscillation_count: 0,
        }
    }

    /// Check if oscillation was detected (count exceeds threshold).
    pub fn is_oscillating(&self) -> bool {
        self.oscillation_count > super::oscillation::DEFAULT_OSCILLATION_THRESHOLD
    }
}

/// Convert a 6-DOF pose vector [tx, ty, tz, roll, pitch, yaw] to an Isometry3.
pub fn pose_vector_to_isometry(pose: &[f64; 6]) -> Isometry3<f64> {
    let translation = Vector3::new(pose[0], pose[1], pose[2]);
    Isometry3::from_parts(translation.into(), rotation_from_pose_angles(pose[3], pose[4], pose[5]))
}

/// Build the rotation a pose vector's angles denote: R = Rx(roll) Ry(pitch) Rz(yaw).
///
/// This is Autoware's convention, and it is what every consumer of a pose
/// vector in this crate already assumes -- `pose_to_transform_matrix`, the GPU
/// kernels' angular derivatives, and `derivatives/cpu.rs`, which builds its
/// rotation in the same XYZ order.
///
/// It is *not* nalgebra's `from_euler_angles`, which composes R = Rz Ry Rx.
/// These converters used to call that, so a pose vector meant one thing at the
/// boundary and another everywhere inside. The two agree only when at most one
/// angle is non-zero, which is why it survived: synthetic tests sit near zero
/// yaw, and the error is common to both the CPU and GPU arms so they agreed
/// with each other while both drifted from the caller's isometry.
pub fn rotation_from_pose_angles(roll: f64, pitch: f64, yaw: f64) -> UnitQuaternion<f64> {
    UnitQuaternion::from_axis_angle(&Vector3::x_axis(), roll)
        * UnitQuaternion::from_axis_angle(&Vector3::y_axis(), pitch)
        * UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw)
}

/// Recover the angles of `rotation_from_pose_angles`, its exact inverse.
///
/// For R = Rx Ry Rz: r02 = sin(pitch), r01 = -cos(pitch) sin(yaw),
/// r00 = cos(pitch) cos(yaw), r12 = -sin(roll) cos(pitch), r22 = cos(roll) cos(pitch).
pub fn pose_angles_from_rotation(rotation: &UnitQuaternion<f64>) -> (f64, f64, f64) {
    let r = rotation.to_rotation_matrix();
    let sp = r[(0, 2)].clamp(-1.0, 1.0);
    let pitch = sp.asin();
    // Near gimbal lock (pitch = +/- 90 deg) roll and yaw are not separable; put
    // the whole rotation in yaw, which is the axis a ground vehicle cares about.
    if (sp.abs() - 1.0).abs() < 1e-9 {
        let yaw = (-r[(1, 0)]).atan2(r[(1, 1)]);
        return (0.0, pitch, yaw);
    }
    let yaw = (-r[(0, 1)]).atan2(r[(0, 0)]);
    let roll = (-r[(1, 2)]).atan2(r[(2, 2)]);
    (roll, pitch, yaw)
}

/// Convert an Isometry3 to a 6-DOF pose vector [tx, ty, tz, roll, pitch, yaw].
pub fn isometry_to_pose_vector(isometry: &Isometry3<f64>) -> [f64; 6] {
    let translation = isometry.translation.vector;
    let (roll, pitch, yaw) = pose_angles_from_rotation(&isometry.rotation);
    [
        translation.x,
        translation.y,
        translation.z,
        roll,
        pitch,
        yaw,
    ]
}

/// Apply a delta vector to a pose, returning the updated pose.
///
/// The delta is in the local frame and is applied as:
/// new_pose = old_pose * exp(delta)
///
/// For small deltas, this is approximately:
/// new_pose ≈ old_pose + delta
pub fn apply_pose_delta(pose: &[f64; 6], delta: &Vector6<f64>, step_size: f64) -> [f64; 6] {
    let scaled_delta = delta * step_size;

    // Apply translation delta
    let new_tx = pose[0] + scaled_delta[0];
    let new_ty = pose[1] + scaled_delta[1];
    let new_tz = pose[2] + scaled_delta[2];

    // Apply rotation delta (Euler angle increment)
    let new_roll = pose[3] + scaled_delta[3];
    let new_pitch = pose[4] + scaled_delta[4];
    let new_yaw = pose[5] + scaled_delta[5];

    [new_tx, new_ty, new_tz, new_roll, new_pitch, new_yaw]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::FRAC_PI_4;

    #[test]
    fn test_config_default() {
        let config = NdtConfig::default();
        assert_eq!(config.resolution, 2.0);
        assert_eq!(config.max_iterations, 30);
        assert!(!config.use_line_search); // Line search disabled by default (matches Autoware)
    }

    #[test]
    fn test_config_with_resolution() {
        let config = NdtConfig::with_resolution(1.0);
        assert_eq!(config.resolution, 1.0);
        assert_eq!(config.max_iterations, 30); // Other defaults preserved
    }

    #[test]
    fn test_convergence_status() {
        assert!(ConvergenceStatus::Converged.is_converged());
        assert!(!ConvergenceStatus::MaxIterations.is_converged());

        assert!(ConvergenceStatus::Converged.is_usable());
        assert!(ConvergenceStatus::MaxIterations.is_usable());
        assert!(!ConvergenceStatus::NoCorrespondences.is_usable());
    }

    /// The pose vector must mean the same rotation at the boundary as it does
    /// inside: `pose_vector_to_isometry` has to agree with the matrix the GPU
    /// and CPU derivative paths build from the same numbers.
    ///
    /// Nothing checked this before, and the two disagreed for any pose with
    /// more than one non-zero angle -- which is every pose a vehicle actually
    /// drives.
    #[test]
    fn test_pose_vector_agrees_with_transform_matrix() {
        use crate::derivatives::gpu::pose_to_transform_matrix;
        let cases = [
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 0.0, 0.0, 3.06],
            [1.0, 2.0, 3.0, 0.05, -0.04, 3.06], // as driven on the COSS bag
            [0.0, 0.0, 0.0, 0.1, 0.2, -2.5],
        ];
        for pose in cases {
            let iso = pose_vector_to_isometry(&pose);
            let m = pose_to_transform_matrix(&pose);
            let expect = iso.to_homogeneous();
            let mut worst = 0.0f64;
            for r in 0..3 {
                for c in 0..4 {
                    worst = worst.max((m[r * 4 + c] as f64 - expect[(r, c)]).abs());
                }
            }
            assert!(
                worst < 1e-6,
                "pose vector {pose:?} means different rotations at the boundary \
                 and inside: differs by {worst:.8}"
            );
        }
    }

    /// The round trip must hold at orientations a vehicle reaches, not just
    /// near zero.
    #[test]
    fn test_pose_vector_roundtrip_at_real_orientations() {
        for pose in [
            [1.0, 2.0, 3.0, 0.05, -0.04, 3.06],
            [0.0, 0.0, 0.0, 0.1, 0.2, -2.5],
            [1.0, -1.0, 0.5, -0.3, 0.4, 1.2],
        ] {
            let recovered = isometry_to_pose_vector(&pose_vector_to_isometry(&pose));
            for i in 0..6 {
                assert_relative_eq!(pose[i], recovered[i], epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_pose_vector_roundtrip() {
        let pose = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3];
        let isometry = pose_vector_to_isometry(&pose);
        let recovered = isometry_to_pose_vector(&isometry);

        for i in 0..6 {
            assert_relative_eq!(pose[i], recovered[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_pose_vector_identity() {
        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let isometry = pose_vector_to_isometry(&pose);

        assert!(isometry.rotation.angle() < 1e-10);
        assert!(isometry.translation.vector.norm() < 1e-10);
    }

    #[test]
    fn test_apply_pose_delta_translation() {
        let pose = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let delta = Vector6::new(0.1, 0.2, 0.3, 0.0, 0.0, 0.0);

        let new_pose = apply_pose_delta(&pose, &delta, 1.0);

        assert_relative_eq!(new_pose[0], 1.1, epsilon = 1e-10);
        assert_relative_eq!(new_pose[1], 2.2, epsilon = 1e-10);
        assert_relative_eq!(new_pose[2], 3.3, epsilon = 1e-10);
    }

    #[test]
    fn test_apply_pose_delta_with_step_size() {
        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let delta = Vector6::new(1.0, 1.0, 1.0, 0.0, 0.0, 0.0);

        let new_pose = apply_pose_delta(&pose, &delta, 0.5);

        assert_relative_eq!(new_pose[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(new_pose[1], 0.5, epsilon = 1e-10);
        assert_relative_eq!(new_pose[2], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_apply_pose_delta_rotation() {
        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let delta = Vector6::new(0.0, 0.0, 0.0, FRAC_PI_4, 0.0, 0.0);

        let new_pose = apply_pose_delta(&pose, &delta, 1.0);

        assert_relative_eq!(new_pose[3], FRAC_PI_4, epsilon = 1e-10);
    }
}
