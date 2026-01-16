//! Newton step computation for NDT optimization.
//!
//! The Newton step solves: Δp = -H⁻¹g
//! where H is the Hessian and g is the gradient.
//!
//! We use SVD for numerical stability when H is near-singular.

use nalgebra::{Matrix6, Vector6};

/// Compute the Newton step using SVD decomposition.
///
/// Solves: Δp = -H⁻¹g using SVD for numerical stability.
///
/// # Arguments
/// * `gradient` - Gradient of the NDT score (6x1)
/// * `hessian` - Hessian of the NDT score (6x6)
/// * `tolerance` - Singular value tolerance for pseudo-inverse
///
/// # Returns
/// The Newton step vector (6x1), or None if Hessian is singular.
pub fn newton_step(
    gradient: &Vector6<f64>,
    hessian: &Matrix6<f64>,
    tolerance: f64,
) -> Option<Vector6<f64>> {
    // Use SVD to solve H * delta = -g
    let svd = hessian.svd(true, true);

    // Check for singular values below tolerance
    let max_sv = svd.singular_values.max();
    if max_sv < tolerance {
        return None; // Hessian is effectively zero
    }

    // Solve using pseudo-inverse with tolerance
    let neg_gradient = -gradient;
    svd.solve(&neg_gradient, tolerance).ok()
}

/// Compute the Newton step with Levenberg-Marquardt regularization.
///
/// Solves: Δp = -(H + λI)⁻¹g
/// where λ is the regularization parameter.
///
/// This provides better numerical stability and prevents oscillation
/// when the Hessian is ill-conditioned.
///
/// # Arguments
/// * `gradient` - Gradient of the NDT score (6x1)
/// * `hessian` - Hessian of the NDT score (6x6)
/// * `regularization` - Regularization parameter λ (typically 1e-6 to 1e-3)
/// * `tolerance` - Singular value tolerance for pseudo-inverse
///
/// # Returns
/// The regularized Newton step vector (6x1), or None if still singular.
pub fn newton_step_regularized(
    gradient: &Vector6<f64>,
    hessian: &Matrix6<f64>,
    regularization: f64,
    tolerance: f64,
) -> Option<Vector6<f64>> {
    // Add regularization to diagonal: H + λI
    let regularized_hessian = hessian + Matrix6::identity() * regularization;

    newton_step(gradient, &regularized_hessian, tolerance)
}

/// Compute the Newton step using Cholesky decomposition.
///
/// This is faster than SVD but requires the Hessian to be positive definite.
/// Falls back to SVD if Cholesky fails.
///
/// # Arguments
/// * `gradient` - Gradient of the NDT score (6x1)
/// * `hessian` - Hessian of the NDT score (6x6)
/// * `tolerance` - Singular value tolerance (used for SVD fallback)
///
/// # Returns
/// The Newton step vector (6x1), or None if Hessian is singular.
pub fn newton_step_cholesky(
    gradient: &Vector6<f64>,
    hessian: &Matrix6<f64>,
    tolerance: f64,
) -> Option<Vector6<f64>> {
    // Try Cholesky decomposition first (requires positive definite)
    if let Some(chol) = hessian.cholesky() {
        let neg_gradient = -gradient;
        return Some(chol.solve(&neg_gradient));
    }

    // Fall back to SVD
    newton_step(gradient, hessian, tolerance)
}

/// Check if a matrix is positive definite by attempting Cholesky decomposition.
pub fn is_positive_definite(matrix: &Matrix6<f64>) -> bool {
    matrix.cholesky().is_some()
}

/// Check if a matrix is negative definite (all eigenvalues < 0).
pub fn is_negative_definite(matrix: &Matrix6<f64>) -> bool {
    let eigen = matrix.symmetric_eigen();
    eigen.eigenvalues.iter().all(|&ev| ev < 0.0)
}

/// Regularize a Hessian to ensure negative definiteness for maximization.
///
/// For NDT score maximization, the Hessian should be negative definite at the maximum.
/// When the Hessian has non-negative eigenvalues (indicating we're not at a maximum
/// or there's numerical issues), this function clamps those eigenvalues to ensure
/// the matrix is negative definite.
///
/// # Arguments
/// * `hessian` - The 6x6 Hessian matrix
/// * `min_eigenvalue` - The minimum (most negative) eigenvalue to enforce (e.g., -1e-3)
///
/// # Returns
/// The regularized Hessian with all eigenvalues <= min_eigenvalue
pub fn regularize_hessian_negative_definite(
    hessian: &Matrix6<f64>,
    min_eigenvalue: f64,
) -> Matrix6<f64> {
    let eigen = hessian.symmetric_eigen();
    let mut eigenvalues = eigen.eigenvalues;

    // Clamp any eigenvalue > min_eigenvalue to min_eigenvalue
    // This ensures all eigenvalues are <= min_eigenvalue (i.e., sufficiently negative)
    let mut modified = false;
    for ev in eigenvalues.iter_mut() {
        if *ev > min_eigenvalue {
            *ev = min_eigenvalue;
            modified = true;
        }
    }

    if !modified {
        return *hessian;
    }

    // Reconstruct: H_reg = V * D_reg * V^T
    let eigenvectors = &eigen.eigenvectors;
    let diag = Matrix6::from_diagonal(&eigenvalues);
    eigenvectors * diag * eigenvectors.transpose()
}

/// Compute Newton step with Hessian regularization for maximization.
///
/// This function ensures the Hessian is negative definite before computing
/// the Newton step. For NDT score maximization, this prevents the Newton
/// step from pointing in the wrong direction when the Hessian is indefinite.
///
/// # Arguments
/// * `gradient` - Gradient of the NDT score (6x1)
/// * `hessian` - Hessian of the NDT score (6x6)
/// * `min_eigenvalue` - Minimum eigenvalue for regularization (e.g., -1e-3)
/// * `tolerance` - Singular value tolerance for SVD fallback
///
/// # Returns
/// Tuple of (Newton step, whether Hessian was regularized)
pub fn newton_step_negative_definite(
    gradient: &Vector6<f64>,
    hessian: &Matrix6<f64>,
    min_eigenvalue: f64,
    tolerance: f64,
) -> (Option<Vector6<f64>>, bool) {
    // Check if Hessian is already negative definite
    if is_negative_definite(hessian) {
        let step = newton_step(gradient, hessian, tolerance);
        return (step, false);
    }

    // Regularize to ensure negative definiteness
    let regularized = regularize_hessian_negative_definite(hessian, min_eigenvalue);
    let step = newton_step(gradient, &regularized, tolerance);
    (step, true)
}

/// Compute the condition number of the Hessian.
///
/// A high condition number indicates the Hessian is ill-conditioned
/// and the Newton step may be unreliable.
///
/// # Returns
/// The condition number (ratio of largest to smallest singular value),
/// or f64::INFINITY if the smallest singular value is zero.
pub fn condition_number(hessian: &Matrix6<f64>) -> f64 {
    let svd = hessian.svd(false, false);
    let singular_values = &svd.singular_values;

    let max_sv = singular_values.max();
    let min_sv = singular_values.min();

    if min_sv < 1e-15 {
        f64::INFINITY
    } else {
        max_sv / min_sv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_newton_step_identity() {
        // For H = I, delta = -g
        let gradient = Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let hessian = Matrix6::identity();

        let delta = newton_step(&gradient, &hessian, 1e-10).unwrap();

        for i in 0..6 {
            assert_relative_eq!(delta[i], -gradient[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_newton_step_scaled() {
        // For H = 2I, delta = -g/2
        let gradient = Vector6::new(2.0, 4.0, 6.0, 8.0, 10.0, 12.0);
        let hessian = Matrix6::identity() * 2.0;

        let delta = newton_step(&gradient, &hessian, 1e-10).unwrap();

        for i in 0..6 {
            assert_relative_eq!(delta[i], -gradient[i] / 2.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_newton_step_zero_gradient() {
        // Zero gradient should give zero step
        let gradient = Vector6::zeros();
        let hessian = Matrix6::identity();

        let delta = newton_step(&gradient, &hessian, 1e-10).unwrap();

        assert!(delta.norm() < 1e-15);
    }

    #[test]
    fn test_newton_step_singular_hessian() {
        // Singular Hessian (all zeros) should return None
        let gradient = Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let hessian = Matrix6::zeros();

        let result = newton_step(&gradient, &hessian, 1e-10);

        assert!(result.is_none());
    }

    #[test]
    fn test_newton_step_regularized() {
        // Regularization should help with ill-conditioned Hessian
        let gradient = Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let mut hessian = Matrix6::identity();
        hessian[(5, 5)] = 1e-12; // Make one eigenvalue very small

        // Without regularization, condition number is huge
        assert!(condition_number(&hessian) > 1e10);

        // With regularization, we should get a valid step
        let delta = newton_step_regularized(&gradient, &hessian, 1e-6, 1e-10);
        assert!(delta.is_some());
    }

    #[test]
    fn test_newton_step_cholesky() {
        // Positive definite Hessian should use Cholesky
        let gradient = Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let hessian = Matrix6::identity() * 2.0; // Positive definite

        let delta = newton_step_cholesky(&gradient, &hessian, 1e-10).unwrap();

        for i in 0..6 {
            assert_relative_eq!(delta[i], -gradient[i] / 2.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_newton_step_cholesky_fallback() {
        // Non-positive-definite should fall back to SVD
        let gradient = Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let mut hessian = Matrix6::identity();
        hessian[(0, 0)] = -1.0; // Not positive definite

        // Should still work via SVD fallback
        let delta = newton_step_cholesky(&gradient, &hessian, 1e-10);
        assert!(delta.is_some());
    }

    #[test]
    fn test_is_positive_definite() {
        assert!(is_positive_definite(&Matrix6::identity()));
        assert!(is_positive_definite(&(Matrix6::identity() * 5.0)));

        let mut not_pd = Matrix6::identity();
        not_pd[(0, 0)] = -1.0;
        assert!(!is_positive_definite(&not_pd));
    }

    #[test]
    fn test_condition_number() {
        // Identity has condition number 1
        assert_relative_eq!(condition_number(&Matrix6::identity()), 1.0, epsilon = 1e-10);

        // Scaled identity still has condition number 1
        assert_relative_eq!(
            condition_number(&(Matrix6::identity() * 10.0)),
            1.0,
            epsilon = 1e-10
        );

        // Ill-conditioned matrix
        let mut ill_cond = Matrix6::identity();
        ill_cond[(0, 0)] = 1e6;
        assert_relative_eq!(condition_number(&ill_cond), 1e6, epsilon = 1e3);
    }

    #[test]
    fn test_newton_step_general_hessian() {
        // Test with a general symmetric positive definite Hessian
        let gradient = Vector6::new(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);

        // Create a positive definite Hessian: H = A^T * A + I
        let a = Matrix6::new(
            1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 2.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 1.5, 0.1, 0.0,
            0.0, 0.0, 0.0, 0.1, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0,
            0.1, 1.0,
        );
        let hessian = a.transpose() * a + Matrix6::identity();

        let delta = newton_step(&gradient, &hessian, 1e-10).unwrap();

        // Verify: H * delta ≈ -gradient
        let result = hessian * delta;
        for i in 0..6 {
            assert_relative_eq!(result[i], -gradient[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_is_negative_definite() {
        // Negative identity is negative definite
        assert!(is_negative_definite(&(-Matrix6::identity())));
        assert!(is_negative_definite(&(-Matrix6::identity() * 5.0)));

        // Positive identity is NOT negative definite
        assert!(!is_negative_definite(&Matrix6::identity()));

        // Mixed eigenvalues is NOT negative definite
        let mut mixed = -Matrix6::identity();
        mixed[(0, 0)] = 1.0; // One positive eigenvalue
        assert!(!is_negative_definite(&mixed));
    }

    #[test]
    fn test_regularize_hessian_negative_definite() {
        // Already negative definite - should return unchanged
        let neg_def = -Matrix6::identity();
        let regularized = regularize_hessian_negative_definite(&neg_def, -1e-3);
        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(regularized[(i, j)], neg_def[(i, j)], epsilon = 1e-10);
            }
        }

        // Mixed eigenvalues - should clamp positive ones
        let mut mixed = -Matrix6::identity();
        mixed[(0, 0)] = 1.0; // One positive eigenvalue
        assert!(!is_negative_definite(&mixed));

        let regularized = regularize_hessian_negative_definite(&mixed, -1e-3);
        assert!(is_negative_definite(&regularized));

        // Check eigenvalues are all <= -1e-3
        let eigen = regularized.symmetric_eigen();
        for ev in eigen.eigenvalues.iter() {
            assert!(*ev <= -1e-3 + 1e-10); // Allow small numerical tolerance
        }
    }

    #[test]
    fn test_newton_step_negative_definite() {
        let gradient = Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);

        // Negative definite Hessian - should not regularize
        let neg_def = -Matrix6::identity() * 2.0;
        let (step, was_regularized) =
            newton_step_negative_definite(&gradient, &neg_def, -1e-3, 1e-10);
        assert!(step.is_some());
        assert!(!was_regularized);

        // Mixed eigenvalues - should regularize
        let mut mixed = -Matrix6::identity();
        mixed[(0, 0)] = 1.0; // One positive eigenvalue
        let (step, was_regularized) =
            newton_step_negative_definite(&gradient, &mixed, -1e-3, 1e-10);
        assert!(step.is_some());
        assert!(was_regularized);
    }
}
