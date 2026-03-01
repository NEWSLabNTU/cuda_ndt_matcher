//! GPU-accelerated NDT derivative computation using CubeCL.
//!
//! This module provides CUDA-accelerated computation of NDT score, gradient, and Hessian.
//! The key algorithm:
//! 1. Transform source points using current pose
//! 2. For each transformed point, find nearby voxels (radius search)
//! 3. Compute per-point-voxel derivatives
//! 4. Reduce across all contributions
//!
//! # GPU Memory Layout
//!
//! - Source points: [N, 3] flattened to [N * 3]
//! - Voxel means: [V, 3] flattened to [V * 3]
//! - Voxel inv_covariances: [V, 9] flattened to [V * 9] (row-major 3x3)
//! - Morton codes: [V] for radius search
//!
//! # Multi-Voxel Radius Search
//!
//! Unlike single-voxel lookup, we search for ALL voxels within a radius of
//! each transformed point. This matches Autoware's `radiusSearch` behavior
//! and provides smoother gradients near voxel boundaries.

use cubecl::prelude::*;

/// Maximum number of neighboring voxels per query point.
/// Autoware typically finds 1-7 neighbors with radius = resolution.
pub const MAX_NEIGHBORS: u32 = 8;

// Note: Helper functions like distance_squared are inlined directly
// in kernels due to CubeCL type inference limitations with generics.

/// Transform a point by a 4x4 transformation matrix.
///
/// # Arguments
/// * `px`, `py`, `pz` - Input point
/// * `transform` - 4x4 transformation matrix (row-major)
///
/// # Returns
/// Transformed point (tx, ty, tz)
#[cube]
fn transform_point_inline<F: Float>(px: F, py: F, pz: F, transform: &Array<F>) -> (F, F, F) {
    let tx = transform[0] * px + transform[1] * py + transform[2] * pz + transform[3];
    let ty = transform[4] * px + transform[5] * py + transform[6] * pz + transform[7];
    let tz = transform[8] * px + transform[9] * py + transform[10] * pz + transform[11];
    (tx, ty, tz)
}

/// Brute-force radius search on GPU.
///
/// For each query point, finds up to MAX_NEIGHBORS voxels within the radius.
/// Outputs neighbor indices (-1 for no neighbor).
///
/// This is O(N*V) but with good GPU parallelism for moderate V.
/// For large V, Morton-based search would be more efficient.
#[cube(launch_unchecked)]
pub fn radius_search_kernel<F: Float>(
    // Query points (transformed source points) [N * 3]
    query_points: &Array<F>,
    // Voxel means [V * 3]
    voxel_means: &Array<F>,
    // Voxel validity flags [V]
    voxel_valid: &Array<u32>,
    // Search radius squared
    radius_sq: F,
    // Number of query points
    num_queries: u32,
    // Number of voxels
    num_voxels: u32,
    // Output: neighbor indices [N * MAX_NEIGHBORS], -1 for no neighbor
    neighbor_indices: &mut Array<i32>,
    // Output: neighbor count per query [N]
    neighbor_counts: &mut Array<u32>,
) {
    let query_idx = ABSOLUTE_POS;

    if query_idx >= num_queries {
        terminate!();
    }

    // Load query point
    let base = query_idx * 3;
    let qx = query_points[base];
    let qy = query_points[base + 1];
    let qz = query_points[base + 2];

    // Initialize neighbor output
    let out_base = query_idx * MAX_NEIGHBORS;
    for i in 0..MAX_NEIGHBORS {
        neighbor_indices[out_base + i] = -1_i32;
    }

    let mut count = 0u32;

    // Search all voxels
    // NOTE: We avoid using `break` here because it triggers a CubeCL optimizer bug
    // in uniformity analysis ("no entry found for key"). Instead, we use a
    // conditional flag to skip processing once we've found enough neighbors.
    for v in 0..num_voxels {
        // Only process if we haven't reached MAX_NEIGHBORS yet
        let should_process = count < MAX_NEIGHBORS;

        if should_process {
            // Skip invalid voxels (use conditional instead of continue)
            let is_valid = voxel_valid[v];
            if is_valid != 0u32 {
                let vbase = v * 3;
                let vx = voxel_means[vbase];
                let vy = voxel_means[vbase + 1];
                let vz = voxel_means[vbase + 2];

                // Inline distance calculation (CubeCL type inference issue with helpers)
                let dx = qx - vx;
                let dy = qy - vy;
                let dz = qz - vz;
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq <= radius_sq {
                    neighbor_indices[out_base + count] = v as i32;
                    count += 1u32;
                }
            }
        }
    }

    neighbor_counts[query_idx] = count;
}

/// Compute NDT score for all point-voxel pairs.
///
/// For each source point, computes score contributions from all neighboring voxels.
///
/// # Algorithm (per point)
/// 1. Transform point using current pose
/// 2. Look up neighbor voxels from pre-computed indices
/// 3. For each neighbor voxel:
///    a. Compute x_trans = transformed_point - voxel_mean
///    b. Compute score contribution: -d1 * exp(-d2/2 * x'Σ⁻¹x)
/// 4. Accumulate to per-point output
#[cube(launch_unchecked)]
pub fn compute_ndt_score_kernel<F: Float>(
    // Source points [N * 3]
    source_points: &Array<F>,
    // Transformation matrix [16]
    transform: &Array<F>,
    // Voxel means [V * 3]
    voxel_means: &Array<F>,
    // Voxel inverse covariances [V * 9] (row-major 3x3)
    voxel_inv_covs: &Array<F>,
    // Neighbor indices from radius search [N * MAX_NEIGHBORS]
    neighbor_indices: &Array<i32>,
    // Neighbor counts [N]
    neighbor_counts: &Array<u32>,
    // Gaussian d1 parameter
    gauss_d1: F,
    // Gaussian d2 parameter
    gauss_d2: F,
    // Number of source points
    num_points: u32,
    // Output: per-point scores [N]
    scores: &mut Array<F>,
    // Output: per-point correspondence counts [N]
    correspondences: &mut Array<u32>,
) {
    let point_idx = ABSOLUTE_POS;

    if point_idx >= num_points {
        terminate!();
    }

    // Load and transform source point
    let sbase = point_idx * 3;
    let sx = source_points[sbase];
    let sy = source_points[sbase + 1];
    let sz = source_points[sbase + 2];

    let (tx, ty, tz) = transform_point_inline(sx, sy, sz, transform);

    // Accumulate score across all neighbors
    let mut total_score = F::new(0.0);
    let mut total_correspondences = 0u32;

    let num_neighbors = neighbor_counts[point_idx];
    let neighbor_base = point_idx * MAX_NEIGHBORS;

    for i in 0..MAX_NEIGHBORS {
        // Only process if within neighbor count
        if i < num_neighbors {
            let voxel_idx = neighbor_indices[neighbor_base + i];
            if voxel_idx >= 0 {
                let v = voxel_idx as u32;

                // Load voxel mean
                let vbase = v * 3;
                let mx = voxel_means[vbase];
                let my = voxel_means[vbase + 1];
                let mz = voxel_means[vbase + 2];

                // Compute x_trans = transformed - mean
                let x0 = tx - mx;
                let x1 = ty - my;
                let x2 = tz - mz;

                // Load inverse covariance (row-major 3x3)
                let cbase = v * 9;
                let c00 = voxel_inv_covs[cbase];
                let c01 = voxel_inv_covs[cbase + 1];
                let c02 = voxel_inv_covs[cbase + 2];
                let c10 = voxel_inv_covs[cbase + 3];
                let c11 = voxel_inv_covs[cbase + 4];
                let c12 = voxel_inv_covs[cbase + 5];
                let c20 = voxel_inv_covs[cbase + 6];
                let c21 = voxel_inv_covs[cbase + 7];
                let c22 = voxel_inv_covs[cbase + 8];

                // Compute x' * Σ⁻¹ * x
                // First: Σ⁻¹ * x
                let cx0 = c00 * x0 + c01 * x1 + c02 * x2;
                let cx1 = c10 * x0 + c11 * x1 + c12 * x2;
                let cx2 = c20 * x0 + c21 * x1 + c22 * x2;

                // Then: x' * (Σ⁻¹ * x)
                let x_c_x = x0 * cx0 + x1 * cx1 + x2 * cx2;

                // Score: -d1 * exp(-d2/2 * x'Σ⁻¹x)
                let exponent = gauss_d2 * x_c_x * F::new(-0.5);
                let score = gauss_d1 * F::new(-1.0) * F::exp(exponent);

                total_score += score;
                total_correspondences += 1u32;
            }
        }
    }

    scores[point_idx] = total_score;
    correspondences[point_idx] = total_correspondences;
}

/// Compute NVTL (Nearest Voxel Transformation Likelihood) scores.
///
/// For each source point, computes the **maximum** score across all neighboring voxels.
/// This matches Autoware's NVTL algorithm where NVTL = average of max scores.
///
/// # Algorithm (per point)
/// 1. Transform point using current pose
/// 2. Look up neighbor voxels from pre-computed indices
/// 3. For each neighbor voxel:
///    a. Compute x_trans = transformed_point - voxel_mean
///    b. Compute score: -d1 * exp(-d2/2 * x'Σ⁻¹x)
/// 4. Track the **maximum** score (not sum)
///
/// # Difference from compute_ndt_score_kernel
/// - `compute_ndt_score_kernel`: Sums scores → for transform probability
/// - `compute_ndt_nvtl_kernel`: Takes max score → for NVTL (Autoware-compatible)
#[cube(launch_unchecked)]
pub fn compute_ndt_nvtl_kernel<F: Float>(
    // Source points [N * 3]
    source_points: &Array<F>,
    // Transformation matrix [16]
    transform: &Array<F>,
    // Voxel means [V * 3]
    voxel_means: &Array<F>,
    // Voxel inverse covariances [V * 9] (row-major 3x3)
    voxel_inv_covs: &Array<F>,
    // Neighbor indices from radius search [N * MAX_NEIGHBORS]
    neighbor_indices: &Array<i32>,
    // Neighbor counts [N]
    neighbor_counts: &Array<u32>,
    // Gaussian d1 parameter
    gauss_d1: F,
    // Gaussian d2 parameter
    gauss_d2: F,
    // Number of source points
    num_points: u32,
    // Output: max score per point [N]
    max_scores: &mut Array<F>,
    // Output: 1 if point has at least one neighbor, 0 otherwise [N]
    has_neighbor: &mut Array<u32>,
) {
    let point_idx = ABSOLUTE_POS;

    if point_idx >= num_points {
        terminate!();
    }

    // Load and transform source point
    let sbase = point_idx * 3;
    let sx = source_points[sbase];
    let sy = source_points[sbase + 1];
    let sz = source_points[sbase + 2];

    let (tx, ty, tz) = transform_point_inline(sx, sy, sz, transform);

    // Track maximum score across all neighbors (NVTL algorithm)
    let mut max_score = F::new(0.0);
    let mut found_neighbor = 0u32;

    let num_neighbors = neighbor_counts[point_idx];
    let neighbor_base = point_idx * MAX_NEIGHBORS;

    for i in 0..MAX_NEIGHBORS {
        // Only process if within neighbor count
        if i < num_neighbors {
            let voxel_idx = neighbor_indices[neighbor_base + i];
            if voxel_idx >= 0 {
                let v = voxel_idx as u32;

                // Load voxel mean
                let vbase = v * 3;
                let mx = voxel_means[vbase];
                let my = voxel_means[vbase + 1];
                let mz = voxel_means[vbase + 2];

                // Compute x_trans = transformed - mean
                let x0 = tx - mx;
                let x1 = ty - my;
                let x2 = tz - mz;

                // Load inverse covariance (row-major 3x3)
                let cbase = v * 9;
                let c00 = voxel_inv_covs[cbase];
                let c01 = voxel_inv_covs[cbase + 1];
                let c02 = voxel_inv_covs[cbase + 2];
                let c10 = voxel_inv_covs[cbase + 3];
                let c11 = voxel_inv_covs[cbase + 4];
                let c12 = voxel_inv_covs[cbase + 5];
                let c20 = voxel_inv_covs[cbase + 6];
                let c21 = voxel_inv_covs[cbase + 7];
                let c22 = voxel_inv_covs[cbase + 8];

                // Compute x' * Σ⁻¹ * x
                // First: Σ⁻¹ * x
                let cx0 = c00 * x0 + c01 * x1 + c02 * x2;
                let cx1 = c10 * x0 + c11 * x1 + c12 * x2;
                let cx2 = c20 * x0 + c21 * x1 + c22 * x2;

                // Then: x' * (Σ⁻¹ * x)
                let x_c_x = x0 * cx0 + x1 * cx1 + x2 * cx2;

                // Score: -d1 * exp(-d2/2 * x'Σ⁻¹x)
                let exponent = gauss_d2 * x_c_x * F::new(-0.5);
                let score = gauss_d1 * F::new(-1.0) * F::exp(exponent);

                // Track maximum score (key difference from sum-based kernel)
                if found_neighbor == 0u32 || score > max_score {
                    max_score = score;
                }
                found_neighbor = 1u32;
            }
        }
    }

    max_scores[point_idx] = max_score;
    has_neighbor[point_idx] = found_neighbor;
}

/// GPU derivative computation context.
///
/// Manages GPU buffers and provides the main API for computing
/// NDT derivatives on GPU.
pub struct GpuDerivatives {
    /// Search radius for voxel lookup.
    pub search_radius: f32,
    /// Gaussian d1 parameter.
    pub gauss_d1: f32,
    /// Gaussian d2 parameter.
    pub gauss_d2: f32,
}

impl GpuDerivatives {
    /// Create a new GPU derivatives context.
    pub fn new(resolution: f64, outlier_ratio: f64) -> Self {
        // Compute Gaussian parameters (same as CPU GaussianParams)
        let gauss_c1 = 10.0 * (1.0 - outlier_ratio);
        let gauss_c2 = outlier_ratio / (resolution * resolution * resolution);
        let gauss_d3 = -gauss_c2.ln();
        let gauss_d1 = -(gauss_c1 + gauss_c2).ln() - gauss_d3;
        let gauss_d2_nom = -(gauss_c1 * (-0.5_f64).exp() + gauss_c2).ln() - gauss_d3;
        let gauss_d2 = -2.0 * (gauss_d2_nom / gauss_d1).ln();

        Self {
            search_radius: resolution as f32,
            gauss_d1: gauss_d1 as f32,
            gauss_d2: gauss_d2 as f32,
        }
    }
}

// ============================================================================
// GPU Runtime Integration
// ============================================================================
//
// The GPU integration requires properly initializing a CUDA runtime and
// managing GPU memory. This is done lazily when `compute_derivatives_gpu`
// is called.
//
// For now, we provide the kernel definitions above. The actual runtime
// integration will be added when we have a working CUDA environment to test.

/// GPU voxel data prepared for derivative computation.
#[derive(Debug, Clone)]
pub struct GpuVoxelData {
    /// Flattened means [V * 3]
    pub means: Vec<f32>,
    /// Flattened inverse covariances [V * 9]
    pub inv_covariances: Vec<f32>,
    /// Flattened principal axes [V * 3] (surface normals for point-to-plane)
    pub principal_axes: Vec<f32>,
    /// Validity flags [V]
    pub valid: Vec<u32>,
    /// Number of voxels
    pub num_voxels: usize,
}

impl GpuVoxelData {
    /// Create GPU voxel data from a VoxelGrid.
    pub fn from_voxel_grid(grid: &crate::voxel_grid::VoxelGrid) -> Self {
        let means = grid.means_flat();
        let inv_covariances = grid.inv_covariances_flat();
        let principal_axes = grid.principal_axes_flat();
        let valid: Vec<u32> = grid.voxels().iter().map(|_| 1u32).collect();
        let num_voxels = grid.len();

        Self {
            means,
            inv_covariances,
            principal_axes,
            valid,
            num_voxels,
        }
    }
}

/// Aggregated GPU derivative result.
#[derive(Debug, Clone)]
pub struct GpuDerivativeResult {
    /// Total score (sum across all points and voxels)
    pub score: f64,
    /// Gradient (6 elements) - sum across all points
    pub gradient: [f64; 6],
    /// Hessian (6x6 matrix) - sum across all points, row-major
    pub hessian: [[f64; 6]; 6],
    /// Number of point-voxel correspondences
    pub num_correspondences: usize,
}

/// Convert pose [tx, ty, tz, roll, pitch, yaw] to 4x4 transformation matrix.
pub fn pose_to_transform_matrix(pose: &[f64; 6]) -> [f32; 16] {
    let (tx, ty, tz) = (pose[0], pose[1], pose[2]);
    let (roll, pitch, yaw) = (pose[3], pose[4], pose[5]);

    let (sr, cr) = roll.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sy, cy) = yaw.sin_cos();

    // Rotation matrix R = Rx(roll) * Ry(pitch) * Rz(yaw)
    // This matches Autoware's convention used in:
    // - eulerAngles(0, 1, 2) extraction
    // - AngleAxis composition: Translation * Rx * Ry * Rz
    // - Jacobian/Hessian angular derivatives (j_ang, h_ang)
    let r00 = cp * cy;
    let r01 = -cp * sy;
    let r02 = sp;
    let r10 = sr * sp * cy + cr * sy;
    let r11 = cr * cy - sr * sp * sy;
    let r12 = -sr * cp;
    let r20 = sr * sy - cr * sp * cy;
    let r21 = cr * sp * sy + sr * cy;
    let r22 = cr * cp;

    // Row-major 4x4 matrix
    [
        r00 as f32, r01 as f32, r02 as f32, tx as f32, r10 as f32, r11 as f32, r12 as f32,
        ty as f32, r20 as f32, r21 as f32, r22 as f32, tz as f32, 0.0, 0.0, 0.0, 1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_derivatives_params() {
        let ctx = GpuDerivatives::new(2.0, 0.55);

        // d1 should be negative, d2 positive
        assert!(ctx.gauss_d1 < 0.0);
        assert!(ctx.gauss_d2 > 0.0);
        assert_eq!(ctx.search_radius, 2.0);
    }
}
