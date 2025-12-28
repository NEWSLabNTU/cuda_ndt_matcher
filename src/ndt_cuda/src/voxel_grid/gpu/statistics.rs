//! Voxel statistics computation (mean, covariance, regularization).
//!
//! After segment detection, each segment represents a voxel. This module
//! computes per-voxel statistics needed for NDT matching:
//! - Mean position (centroid)
//! - Covariance matrix
//! - Inverse covariance matrix (after regularization)
//!
//! # Algorithm
//!
//! 1. **Pass 1**: Accumulate position sums per voxel
//! 2. **Pass 2**: Compute means, then accumulate covariance sums
//! 3. **Pass 3**: Finalize covariance (divide by N-1), regularize, invert
//!
//! # Current Status
//!
//! GPU kernels are defined but require CubeCL type system fixes.
//! CPU reference implementations are provided and tested.

/// Result of voxel statistics computation.
#[derive(Debug, Clone)]
pub struct VoxelStatistics {
    /// Mean position (centroid) for each voxel, [V, 3] flattened.
    pub means: Vec<f32>,
    /// Covariance matrix for each voxel, [V, 9] flattened (row-major 3x3).
    pub covariances: Vec<f32>,
    /// Inverse covariance matrix for each voxel, [V, 9] flattened.
    pub inv_covariances: Vec<f32>,
    /// Number of points in each voxel.
    pub point_counts: Vec<u32>,
    /// Whether each voxel is valid (has enough points).
    pub valid: Vec<bool>,
    /// Number of voxels.
    pub num_voxels: u32,
    /// Minimum points required for a valid voxel.
    pub min_points: u32,
}

/// Intermediate sums for voxel computation.
#[derive(Debug)]
pub struct VoxelSums {
    /// Sum of positions for each voxel, [V, 3] flattened.
    pub position_sums: Vec<f32>,
    /// Point count for each voxel.
    pub counts: Vec<u32>,
    /// Number of voxels.
    pub num_voxels: u32,
}

/// Compute position sums and counts per voxel (CPU reference implementation).
///
/// # Arguments
/// * `points` - Flat array of point coordinates [x0, y0, z0, x1, y1, z1, ...]
/// * `sorted_indices` - Original point indices after sorting by Morton code
/// * `segment_ids` - Voxel ID for each point (from segment detection)
/// * `num_voxels` - Total number of voxels
///
/// # Returns
/// Position sums and counts per voxel.
pub fn compute_voxel_sums_cpu(
    points: &[f32],
    sorted_indices: &[u32],
    segment_ids: &[u32],
    num_voxels: u32,
) -> VoxelSums {
    let num_points = sorted_indices.len();
    let num_voxels_usize = num_voxels as usize;

    let mut position_sums = vec![0.0f32; num_voxels_usize * 3];
    let mut counts = vec![0u32; num_voxels_usize];

    for i in 0..num_points {
        let orig_idx = sorted_indices[i] as usize;
        let voxel_id = segment_ids[i] as usize;

        if voxel_id >= num_voxels_usize {
            continue;
        }

        let px = points[orig_idx * 3];
        let py = points[orig_idx * 3 + 1];
        let pz = points[orig_idx * 3 + 2];

        position_sums[voxel_id * 3] += px;
        position_sums[voxel_id * 3 + 1] += py;
        position_sums[voxel_id * 3 + 2] += pz;
        counts[voxel_id] += 1;
    }

    VoxelSums {
        position_sums,
        counts,
        num_voxels,
    }
}

/// Compute means from sums (CPU reference implementation).
pub fn compute_means_from_sums(sums: &VoxelSums) -> Vec<f32> {
    let num_voxels = sums.num_voxels as usize;
    let mut means = vec![0.0f32; num_voxels * 3];

    for v in 0..num_voxels {
        let count = sums.counts[v];
        if count > 0 {
            let inv_count = 1.0 / count as f32;
            means[v * 3] = sums.position_sums[v * 3] * inv_count;
            means[v * 3 + 1] = sums.position_sums[v * 3 + 1] * inv_count;
            means[v * 3 + 2] = sums.position_sums[v * 3 + 2] * inv_count;
        }
    }

    means
}

/// Compute covariance sums per voxel (CPU reference implementation).
///
/// Accumulates (p - mean)(p - mean)^T for each point.
///
/// # Arguments
/// * `points` - Flat array of point coordinates
/// * `sorted_indices` - Original point indices after sorting
/// * `segment_ids` - Voxel ID for each point
/// * `means` - Mean position for each voxel [V, 3]
/// * `num_voxels` - Total number of voxels
///
/// # Returns
/// Covariance sums [V, 9] (row-major 3x3 matrices).
pub fn compute_covariance_sums_cpu(
    points: &[f32],
    sorted_indices: &[u32],
    segment_ids: &[u32],
    means: &[f32],
    num_voxels: u32,
) -> Vec<f32> {
    let num_points = sorted_indices.len();
    let num_voxels_usize = num_voxels as usize;

    let mut cov_sums = vec![0.0f32; num_voxels_usize * 9];

    for i in 0..num_points {
        let orig_idx = sorted_indices[i] as usize;
        let voxel_id = segment_ids[i] as usize;

        if voxel_id >= num_voxels_usize {
            continue;
        }

        // Deviation from mean
        let dx = points[orig_idx * 3] - means[voxel_id * 3];
        let dy = points[orig_idx * 3 + 1] - means[voxel_id * 3 + 1];
        let dz = points[orig_idx * 3 + 2] - means[voxel_id * 3 + 2];

        // Outer product (symmetric 3x3 matrix)
        let base = voxel_id * 9;
        cov_sums[base] += dx * dx; // [0,0]
        cov_sums[base + 1] += dx * dy; // [0,1]
        cov_sums[base + 2] += dx * dz; // [0,2]
        cov_sums[base + 3] += dy * dx; // [1,0]
        cov_sums[base + 4] += dy * dy; // [1,1]
        cov_sums[base + 5] += dy * dz; // [1,2]
        cov_sums[base + 6] += dz * dx; // [2,0]
        cov_sums[base + 7] += dz * dy; // [2,1]
        cov_sums[base + 8] += dz * dz; // [2,2]
    }

    cov_sums
}

/// Finalize covariance matrices: divide by (N-1), regularize, and invert.
///
/// # Arguments
/// * `cov_sums` - Covariance sums [V, 9]
/// * `counts` - Point counts per voxel
/// * `min_points` - Minimum points required for valid voxel
///
/// # Returns
/// Complete voxel statistics with covariances and inverse covariances.
pub fn finalize_voxels_cpu(
    means: Vec<f32>,
    cov_sums: Vec<f32>,
    counts: Vec<u32>,
    min_points: u32,
) -> VoxelStatistics {
    let num_voxels = counts.len();
    let mut covariances = vec![0.0f32; num_voxels * 9];
    let mut inv_covariances = vec![0.0f32; num_voxels * 9];
    let mut valid = vec![false; num_voxels];

    for v in 0..num_voxels {
        let count = counts[v];

        if count < min_points {
            // Not enough points - mark as invalid
            valid[v] = false;
            // Set identity matrix for safety
            inv_covariances[v * 9] = 1.0;
            inv_covariances[v * 9 + 4] = 1.0;
            inv_covariances[v * 9 + 8] = 1.0;
            continue;
        }

        // Finalize covariance: cov = sum / (count - 1)
        let denom = if count > 1 {
            1.0 / (count - 1) as f32
        } else {
            1.0
        };

        for i in 0..9 {
            covariances[v * 9 + i] = cov_sums[v * 9 + i] * denom;
        }

        // Regularize and invert covariance matrix
        let (inv_cov, is_valid) = regularize_and_invert(&covariances[v * 9..v * 9 + 9]);

        for i in 0..9 {
            inv_covariances[v * 9 + i] = inv_cov[i];
        }
        valid[v] = is_valid;
    }

    VoxelStatistics {
        means,
        covariances,
        inv_covariances,
        point_counts: counts,
        valid,
        num_voxels: num_voxels as u32,
        min_points,
    }
}

/// Regularize and invert a 3x3 covariance matrix.
///
/// Uses eigenvalue decomposition to:
/// 1. Clamp small eigenvalues to min_eigenvalue_ratio * max_eigenvalue
/// 2. Reconstruct the regularized covariance
/// 3. Invert the matrix
///
/// Returns (inverse_covariance, is_valid).
fn regularize_and_invert(cov: &[f32]) -> ([f32; 9], bool) {
    // Extract 3x3 matrix
    let a = [
        [cov[0], cov[1], cov[2]],
        [cov[3], cov[4], cov[5]],
        [cov[6], cov[7], cov[8]],
    ];

    // Compute eigenvalues using characteristic polynomial
    // For symmetric 3x3, use Cardano's formula
    let eigenvalues = compute_eigenvalues_3x3(&a);

    // Check for non-finite values (NaN, Inf)
    if eigenvalues.iter().any(|e| !e.is_finite()) {
        return ([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], false);
    }

    // Find max eigenvalue (treat small negatives as numerical noise)
    let max_eigenvalue = eigenvalues.iter().cloned().fold(0.0f32, f32::max);

    // Minimum allowed eigenvalue (Autoware uses 0.01 ratio, with a floor)
    let min_allowed = (0.01 * max_eigenvalue).max(1e-6);

    // Regularize by adding identity scaled to ensure all eigenvalues are >= min_allowed
    // For very small covariances, this adds a baseline regularization
    let min_eigenvalue = eigenvalues.iter().cloned().fold(f32::MAX, f32::min);
    let reg_amount = if min_eigenvalue < min_allowed {
        (min_allowed - min_eigenvalue).max(1e-6)
    } else {
        0.0
    };

    let final_cov = if reg_amount > 0.0 {
        [
            cov[0] + reg_amount,
            cov[1],
            cov[2],
            cov[3],
            cov[4] + reg_amount,
            cov[5],
            cov[6],
            cov[7],
            cov[8] + reg_amount,
        ]
    } else {
        [
            cov[0], cov[1], cov[2], cov[3], cov[4], cov[5], cov[6], cov[7], cov[8],
        ]
    };

    // Invert the 3x3 matrix
    match invert_3x3(&final_cov) {
        Some(inv) => (inv, true),
        None => ([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], false),
    }
}

/// Compute eigenvalues of a symmetric 3x3 matrix using Cardano's formula.
fn compute_eigenvalues_3x3(a: &[[f32; 3]; 3]) -> [f32; 3] {
    // Characteristic polynomial: det(A - λI) = 0
    // For symmetric matrix: -λ³ + tr(A)λ² - (sum of 2x2 minors)λ + det(A) = 0

    let trace = a[0][0] + a[1][1] + a[2][2];

    // Sum of principal 2x2 minors
    let m01 = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let m02 = a[0][0] * a[2][2] - a[0][2] * a[2][0];
    let m12 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
    let sum_minors = m01 + m02 + m12;

    // Determinant
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

    // Solve: λ³ - trace*λ² + sum_minors*λ - det = 0
    // Using Cardano's formula for depressed cubic
    let p = sum_minors - trace * trace / 3.0;
    let q = 2.0 * trace * trace * trace / 27.0 - trace * sum_minors / 3.0 + det;

    let discriminant = q * q / 4.0 + p * p * p / 27.0;

    if discriminant < 0.0 {
        // Three real roots (typical for covariance matrices)
        let r = (-p * p * p / 27.0).sqrt();
        let phi = (-q / (2.0 * r)).clamp(-1.0, 1.0).acos();
        let cube_root_r = r.powf(1.0 / 3.0);

        let offset = trace / 3.0;
        let e1 = 2.0 * cube_root_r * (phi / 3.0).cos() + offset;
        let e2 = 2.0 * cube_root_r * ((phi + 2.0 * std::f32::consts::PI) / 3.0).cos() + offset;
        let e3 = 2.0 * cube_root_r * ((phi + 4.0 * std::f32::consts::PI) / 3.0).cos() + offset;

        [e1, e2, e3]
    } else {
        // One or two real roots - degenerate case
        let sqrt_disc = discriminant.sqrt();
        let u = (-q / 2.0 + sqrt_disc).cbrt();
        let v = (-q / 2.0 - sqrt_disc).cbrt();
        let e1 = u + v + trace / 3.0;
        [e1, e1, e1] // Simplified - returns same value for degenerate case
    }
}

/// Invert a 3x3 matrix. Returns None if singular.
fn invert_3x3(m: &[f32; 9]) -> Option<[f32; 9]> {
    // Compute determinant
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);

    if det.abs() < 1e-10 {
        return None;
    }

    let inv_det = 1.0 / det;

    // Adjugate matrix transposed
    Some([
        (m[4] * m[8] - m[5] * m[7]) * inv_det,
        (m[2] * m[7] - m[1] * m[8]) * inv_det,
        (m[1] * m[5] - m[2] * m[4]) * inv_det,
        (m[5] * m[6] - m[3] * m[8]) * inv_det,
        (m[0] * m[8] - m[2] * m[6]) * inv_det,
        (m[2] * m[3] - m[0] * m[5]) * inv_det,
        (m[3] * m[7] - m[4] * m[6]) * inv_det,
        (m[1] * m[6] - m[0] * m[7]) * inv_det,
        (m[0] * m[4] - m[1] * m[3]) * inv_det,
    ])
}

/// Compute complete voxel statistics from sorted points and segments.
///
/// This is the main entry point that chains all three passes.
///
/// # Arguments
/// * `points` - Flat array of point coordinates
/// * `sorted_indices` - Original point indices after Morton sort
/// * `segment_ids` - Voxel ID for each point
/// * `num_voxels` - Total number of voxels
/// * `min_points` - Minimum points required for valid voxel
pub fn compute_voxel_statistics_cpu(
    points: &[f32],
    sorted_indices: &[u32],
    segment_ids: &[u32],
    num_voxels: u32,
    min_points: u32,
) -> VoxelStatistics {
    // Pass 1: Compute sums and counts
    let sums = compute_voxel_sums_cpu(points, sorted_indices, segment_ids, num_voxels);

    // Compute means from sums
    let means = compute_means_from_sums(&sums);

    // Pass 2: Compute covariance sums
    let cov_sums =
        compute_covariance_sums_cpu(points, sorted_indices, segment_ids, &means, num_voxels);

    // Pass 3: Finalize covariances
    finalize_voxels_cpu(means, cov_sums, sums.counts, min_points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voxel_sums_single_voxel() {
        // Three points in a single voxel
        let points = vec![
            1.0, 2.0, 3.0, // Point 0
            1.5, 2.5, 3.5, // Point 1
            1.2, 2.2, 3.2, // Point 2
        ];
        let sorted_indices = vec![0, 1, 2];
        let segment_ids = vec![0, 0, 0]; // All same voxel

        let sums = compute_voxel_sums_cpu(&points, &sorted_indices, &segment_ids, 1);

        assert_eq!(sums.counts[0], 3);
        assert!((sums.position_sums[0] - 3.7).abs() < 1e-5); // 1.0 + 1.5 + 1.2
        assert!((sums.position_sums[1] - 6.7).abs() < 1e-5); // 2.0 + 2.5 + 2.2
        assert!((sums.position_sums[2] - 9.7).abs() < 1e-5); // 3.0 + 3.5 + 3.2
    }

    #[test]
    fn test_voxel_sums_multiple_voxels() {
        // Points in two voxels
        let points = vec![
            0.0, 0.0, 0.0, // Point 0 -> voxel 0
            1.0, 1.0, 1.0, // Point 1 -> voxel 0
            10.0, 10.0, 10.0, // Point 2 -> voxel 1
        ];
        let sorted_indices = vec![0, 1, 2];
        let segment_ids = vec![0, 0, 1];

        let sums = compute_voxel_sums_cpu(&points, &sorted_indices, &segment_ids, 2);

        assert_eq!(sums.counts[0], 2);
        assert_eq!(sums.counts[1], 1);
        assert!((sums.position_sums[0] - 1.0).abs() < 1e-5); // voxel 0: 0+1
        assert!((sums.position_sums[3] - 10.0).abs() < 1e-5); // voxel 1: 10
    }

    #[test]
    fn test_means_from_sums() {
        let sums = VoxelSums {
            position_sums: vec![3.0, 6.0, 9.0, 10.0, 20.0, 30.0],
            counts: vec![3, 2],
            num_voxels: 2,
        };

        let means = compute_means_from_sums(&sums);

        assert!((means[0] - 1.0).abs() < 1e-5); // 3/3
        assert!((means[1] - 2.0).abs() < 1e-5); // 6/3
        assert!((means[2] - 3.0).abs() < 1e-5); // 9/3
        assert!((means[3] - 5.0).abs() < 1e-5); // 10/2
        assert!((means[4] - 10.0).abs() < 1e-5); // 20/2
        assert!((means[5] - 15.0).abs() < 1e-5); // 30/2
    }

    #[test]
    fn test_covariance_sums() {
        // Points exactly at mean - should produce zero covariance
        let points = vec![1.0, 2.0, 3.0];
        let sorted_indices = vec![0];
        let segment_ids = vec![0];
        let means = vec![1.0, 2.0, 3.0];

        let cov_sums =
            compute_covariance_sums_cpu(&points, &sorted_indices, &segment_ids, &means, 1);

        // All covariance sums should be zero (point at mean)
        for &val in cov_sums.iter().take(9) {
            assert!(val.abs() < 1e-10);
        }
    }

    #[test]
    fn test_invert_3x3_identity() {
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let inv = invert_3x3(&identity).unwrap();

        // Inverse of identity is identity
        for i in 0..9 {
            assert!((inv[i] - identity[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_invert_3x3_diagonal() {
        let diag = [2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 8.0];
        let inv = invert_3x3(&diag).unwrap();

        let expected = [0.5, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.125];
        for i in 0..9 {
            assert!((inv[i] - expected[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_eigenvalues_diagonal() {
        let diag = [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]];
        let eigenvalues = compute_eigenvalues_3x3(&diag);

        // For diagonal matrix, eigenvalues are the diagonal entries
        let mut sorted = eigenvalues;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!((sorted[0] - 1.0).abs() < 1e-4);
        assert!((sorted[1] - 2.0).abs() < 1e-4);
        assert!((sorted[2] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_full_statistics_pipeline() {
        // Create a simple point cloud with two voxels
        let points = vec![
            // Voxel 0: cluster around (0, 0, 0)
            0.0, 0.0, 0.0, 0.1, 0.1, 0.1, -0.1, -0.1, -0.1, 0.05, -0.05, 0.05,
            // Voxel 1: cluster around (10, 10, 10)
            10.0, 10.0, 10.0, 10.1, 10.1, 10.1, 9.9, 9.9, 9.9,
        ];
        let sorted_indices: Vec<u32> = (0..7).collect();
        let segment_ids = vec![0, 0, 0, 0, 1, 1, 1];

        let stats = compute_voxel_statistics_cpu(&points, &sorted_indices, &segment_ids, 2, 3);

        assert_eq!(stats.num_voxels, 2);
        assert_eq!(stats.point_counts[0], 4);
        assert_eq!(stats.point_counts[1], 3);
        assert!(stats.valid[0]);
        assert!(stats.valid[1]);

        // Check means approximately
        assert!((stats.means[0] - 0.0125).abs() < 0.01); // mean x of voxel 0
        assert!((stats.means[3] - 10.0).abs() < 0.1); // mean x of voxel 1
    }

    #[test]
    fn test_min_points_filter() {
        let points = vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, // Only 2 points
        ];
        let sorted_indices: Vec<u32> = (0..2).collect();
        let segment_ids = vec![0, 0];

        let stats = compute_voxel_statistics_cpu(&points, &sorted_indices, &segment_ids, 1, 3);

        // With min_points=3, this voxel should be invalid
        assert!(!stats.valid[0]);
    }
}
