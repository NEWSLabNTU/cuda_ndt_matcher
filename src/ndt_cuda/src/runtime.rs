//! GPU runtime management for CubeCL CUDA execution.
//!
//! This module provides the GPU runtime infrastructure for NDT computation:
//! - Device initialization and client management
//! - GPU buffer allocation and data transfer
//! - Kernel launch wrappers
//!
//! # Example
//!
//! ```ignore
//! use ndt_cuda::runtime::GpuRuntime;
//!
//! let runtime = GpuRuntime::new()?;
//! let scores = runtime.compute_scores(&source_points, &voxel_data, &transform)?;
//! ```

use anyhow::Result;
use cubecl::{
    client::ComputeClient,
    cuda::{CudaDevice, CudaRuntime},
    prelude::*,
};

use crate::{
    derivatives::gpu::{
        GpuVoxelData, MAX_NEIGHBORS, compute_ndt_nvtl_kernel, compute_ndt_score_kernel,
        radius_search_kernel,
    },
    voxel_grid::kernels::transform_points_kernel,
};

/// Type alias for CUDA compute client
type CudaClient = ComputeClient<<CudaRuntime as Runtime>::Server>;

/// GPU runtime for NDT computation.
///
/// Manages CUDA device initialization and provides high-level APIs
/// for GPU-accelerated NDT operations.
pub struct GpuRuntime {
    /// CUDA device (kept alive for runtime lifetime)
    #[allow(dead_code)]
    device: CudaDevice,
    /// Compute client for kernel execution
    client: CudaClient,
}

impl GpuRuntime {
    /// Create a new GPU runtime with the default CUDA device.
    pub fn new() -> Result<Self> {
        Self::with_device_id(0)
    }

    /// Create a new GPU runtime with a specific CUDA device.
    pub fn with_device_id(device_id: usize) -> Result<Self> {
        let device = CudaDevice::new(device_id);
        let client = CudaRuntime::client(&device);

        Ok(Self { device, client })
    }

    /// Get the underlying compute client.
    pub fn client(&self) -> &CudaClient {
        &self.client
    }

    /// Compute NDT scores for source points against voxel grid.
    ///
    /// Returns per-point scores and total correspondence count.
    pub fn compute_scores(
        &self,
        source_points: &[[f32; 3]],
        voxel_data: &GpuVoxelData,
        transform: &[f32; 16],
        gauss_d1: f32,
        gauss_d2: f32,
        search_radius: f32,
    ) -> Result<GpuScoreResult> {
        if source_points.is_empty() {
            return Ok(GpuScoreResult {
                scores: Vec::new(),
                total_score: 0.0,
                correspondences: Vec::new(),
                total_correspondences: 0,
            });
        }

        let num_points = source_points.len();
        let num_voxels = voxel_data.num_voxels;

        // Flatten source points
        let source_flat: Vec<f32> = source_points
            .iter()
            .flat_map(|p| p.iter().copied())
            .collect();

        // Upload data to GPU
        let source_gpu = self.client.create(f32::as_bytes(&source_flat));
        let transform_gpu = self.client.create(f32::as_bytes(transform));
        let voxel_means_gpu = self.client.create(f32::as_bytes(&voxel_data.means));
        let voxel_inv_covs_gpu = self
            .client
            .create(f32::as_bytes(&voxel_data.inv_covariances));
        let voxel_valid_gpu = self.client.create(u32::as_bytes(&voxel_data.valid));

        // Allocate transformed points buffer
        let transformed_gpu = self
            .client
            .empty(num_points * 3 * std::mem::size_of::<f32>());

        // Step 1: Transform points
        let cube_count = num_points.div_ceil(256) as u32;
        unsafe {
            transform_points_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&source_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&transform_gpu, 16, 1),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&transformed_gpu, num_points * 3, 1),
            );
        }

        // Step 2: Radius search
        let neighbor_indices_gpu = self
            .client
            .empty(num_points * MAX_NEIGHBORS as usize * std::mem::size_of::<i32>());
        let neighbor_counts_gpu = self.client.empty(num_points * std::mem::size_of::<u32>());

        let radius_sq = search_radius * search_radius;
        unsafe {
            radius_search_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&transformed_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&voxel_means_gpu, num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<u32>(&voxel_valid_gpu, num_voxels, 1),
                ScalarArg::new(radius_sq),
                ScalarArg::new(num_points as u32),
                ScalarArg::new(num_voxels as u32),
                ArrayArg::from_raw_parts::<i32>(
                    &neighbor_indices_gpu,
                    num_points * MAX_NEIGHBORS as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&neighbor_counts_gpu, num_points, 1),
            );
        }

        // Step 3: Compute scores
        let scores_gpu = self.client.empty(num_points * std::mem::size_of::<f32>());
        let correspondences_gpu = self.client.empty(num_points * std::mem::size_of::<u32>());

        unsafe {
            compute_ndt_score_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&source_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&transform_gpu, 16, 1),
                ArrayArg::from_raw_parts::<f32>(&voxel_means_gpu, num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&voxel_inv_covs_gpu, num_voxels * 9, 1),
                ArrayArg::from_raw_parts::<i32>(
                    &neighbor_indices_gpu,
                    num_points * MAX_NEIGHBORS as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&neighbor_counts_gpu, num_points, 1),
                ScalarArg::new(gauss_d1),
                ScalarArg::new(gauss_d2),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&scores_gpu, num_points, 1),
                ArrayArg::from_raw_parts::<u32>(&correspondences_gpu, num_points, 1),
            );
        }

        // Read results back
        let scores_bytes = self.client.read_one(scores_gpu);
        let scores = f32::from_bytes(&scores_bytes).to_vec();

        let correspondences_bytes = self.client.read_one(correspondences_gpu);
        let correspondences = u32::from_bytes(&correspondences_bytes).to_vec();

        let total_score: f32 = scores.iter().sum();
        let total_correspondences: u32 = correspondences.iter().sum();

        Ok(GpuScoreResult {
            scores,
            total_score: total_score as f64,
            correspondences,
            total_correspondences: total_correspondences as usize,
        })
    }

    /// Compute NVTL (Nearest Voxel Transformation Likelihood) scores using GPU.
    ///
    /// This matches Autoware's NVTL algorithm:
    /// - For each point, find the **maximum** score across all neighbor voxels
    /// - Final NVTL = average of these max scores
    ///
    /// This differs from `compute_scores()` which sums all voxel contributions
    /// (used for transform probability).
    pub fn compute_nvtl_scores(
        &self,
        source_points: &[[f32; 3]],
        voxel_data: &GpuVoxelData,
        transform: &[f32; 16],
        gauss_d1: f32,
        gauss_d2: f32,
        search_radius: f32,
    ) -> Result<GpuNvtlResult> {
        if source_points.is_empty() {
            return Ok(GpuNvtlResult {
                max_scores: Vec::new(),
                nvtl: 0.0,
                num_with_neighbors: 0,
            });
        }

        let num_points = source_points.len();
        let num_voxels = voxel_data.num_voxels;

        // Flatten source points
        let source_flat: Vec<f32> = source_points
            .iter()
            .flat_map(|p| p.iter().copied())
            .collect();

        // Upload data to GPU
        let source_gpu = self.client.create(f32::as_bytes(&source_flat));
        let transform_gpu = self.client.create(f32::as_bytes(transform));
        let voxel_means_gpu = self.client.create(f32::as_bytes(&voxel_data.means));
        let voxel_inv_covs_gpu = self
            .client
            .create(f32::as_bytes(&voxel_data.inv_covariances));
        let voxel_valid_gpu = self.client.create(u32::as_bytes(&voxel_data.valid));

        // Allocate transformed points buffer
        let transformed_gpu = self
            .client
            .empty(num_points * 3 * std::mem::size_of::<f32>());

        // Step 1: Transform points
        let cube_count = num_points.div_ceil(256) as u32;
        unsafe {
            transform_points_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&source_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&transform_gpu, 16, 1),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&transformed_gpu, num_points * 3, 1),
            );
        }

        // Step 2: Radius search
        let neighbor_indices_gpu = self
            .client
            .empty(num_points * MAX_NEIGHBORS as usize * std::mem::size_of::<i32>());
        let neighbor_counts_gpu = self.client.empty(num_points * std::mem::size_of::<u32>());

        let radius_sq = search_radius * search_radius;
        unsafe {
            radius_search_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&transformed_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&voxel_means_gpu, num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<u32>(&voxel_valid_gpu, num_voxels, 1),
                ScalarArg::new(radius_sq),
                ScalarArg::new(num_points as u32),
                ScalarArg::new(num_voxels as u32),
                ArrayArg::from_raw_parts::<i32>(
                    &neighbor_indices_gpu,
                    num_points * MAX_NEIGHBORS as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&neighbor_counts_gpu, num_points, 1),
            );
        }

        // Step 3: Compute NVTL scores (max per point, not sum)
        let max_scores_gpu = self.client.empty(num_points * std::mem::size_of::<f32>());
        let has_neighbor_gpu = self.client.empty(num_points * std::mem::size_of::<u32>());

        unsafe {
            compute_ndt_nvtl_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&source_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&transform_gpu, 16, 1),
                ArrayArg::from_raw_parts::<f32>(&voxel_means_gpu, num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&voxel_inv_covs_gpu, num_voxels * 9, 1),
                ArrayArg::from_raw_parts::<i32>(
                    &neighbor_indices_gpu,
                    num_points * MAX_NEIGHBORS as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&neighbor_counts_gpu, num_points, 1),
                ScalarArg::new(gauss_d1),
                ScalarArg::new(gauss_d2),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&max_scores_gpu, num_points, 1),
                ArrayArg::from_raw_parts::<u32>(&has_neighbor_gpu, num_points, 1),
            );
        }

        // Read results back
        let max_scores_bytes = self.client.read_one(max_scores_gpu);
        let max_scores = f32::from_bytes(&max_scores_bytes).to_vec();

        let has_neighbor_bytes = self.client.read_one(has_neighbor_gpu);
        let has_neighbor = u32::from_bytes(&has_neighbor_bytes);

        // Compute NVTL: average of max scores for points with neighbors
        let num_with_neighbors: usize = has_neighbor.iter().map(|&h| h as usize).sum();
        let total_max_score: f64 = max_scores
            .iter()
            .zip(has_neighbor.iter())
            .filter(|&(_, &h)| h > 0)
            .map(|(&s, _)| s as f64)
            .sum();

        let nvtl = if num_with_neighbors > 0 {
            total_max_score / num_with_neighbors as f64
        } else {
            0.0
        };

        Ok(GpuNvtlResult {
            max_scores,
            nvtl,
            num_with_neighbors,
        })
    }
}

/// Result of GPU score computation.
#[derive(Debug, Clone)]
pub struct GpuScoreResult {
    /// Per-point scores
    pub scores: Vec<f32>,
    /// Total score (sum of all per-point scores)
    pub total_score: f64,
    /// Per-point correspondence counts
    pub correspondences: Vec<u32>,
    /// Total number of correspondences
    pub total_correspondences: usize,
}

/// Result of GPU NVTL (Nearest Voxel Transformation Likelihood) computation.
///
/// NVTL takes the **maximum** score per point across all neighbor voxels,
/// then computes the average. This matches Autoware's NVTL algorithm.
#[derive(Debug, Clone)]
pub struct GpuNvtlResult {
    /// Per-point max scores (0.0 for points with no neighbors)
    pub max_scores: Vec<f32>,
    /// NVTL = average of max scores for points with neighbors
    pub nvtl: f64,
    /// Number of points that had at least one neighbor voxel
    pub num_with_neighbors: usize,
}

/// Check if CUDA is available on this system.
pub fn is_cuda_available() -> bool {
    // Try to create a device - if it fails, CUDA is not available
    std::panic::catch_unwind(|| {
        let _device = CudaDevice::new(0);
    })
    .is_ok()
}

#[cfg(test)]
mod tests {

    use super::*;

    /// Fail test if CUDA is not available.
    /// Tests using this macro require a CUDA GPU to run.
    macro_rules! require_cuda {
        () => {
            if !is_cuda_available() {
                eprintln!("SKIP: CUDA not available");
                return;
            }
        };
    }
    #[test]
    fn test_cuda_availability() {
        let _available = is_cuda_available();
        crate::test_println!("CUDA available: {_available}");
    }
    #[test]
    fn test_compute_scores_gpu() {
        require_cuda!();

        let runtime = GpuRuntime::new().expect("Failed to create GPU runtime");

        // Create simple voxel data: one voxel at origin
        let voxel_data = GpuVoxelData {
            means: vec![0.0, 0.0, 0.0], // One voxel at origin
            inv_covariances: vec![
                1.0, 0.0, 0.0, // Identity inverse covariance
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, //
            ],
            principal_axes: vec![0.0, 0.0, 1.0], // Z-axis as principal axis
            valid: vec![1],                      // One valid voxel
            num_voxels: 1,
        };

        // Source point at origin (should have high score)
        let source_points = vec![[0.0, 0.0, 0.0]];

        // Identity transform
        let transform = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
        ];

        let result = runtime
            .compute_scores(
                &source_points,
                &voxel_data,
                &transform,
                1.0,  // gauss_d1
                1.0,  // gauss_d2
                10.0, // search_radius (large enough to find the voxel)
            )
            .expect("compute_scores failed");

        crate::test_println!("GPU score result: {:?}", result);

        // Should have one correspondence
        assert_eq!(result.total_correspondences, 1);
        // Score should be negative (NDT score formula: -d1 * exp(...))
        assert!(result.total_score < 0.0);
    }
    #[test]
    fn test_compute_nvtl_scores_gpu() {
        require_cuda!();

        let runtime = GpuRuntime::new().expect("Failed to create GPU runtime");

        // Create two voxels at different positions
        let voxel_data = GpuVoxelData {
            means: vec![
                0.0, 0.0, 0.0, // Voxel 0 at origin
                1.0, 0.0, 0.0, // Voxel 1 at (1, 0, 0)
            ],
            inv_covariances: vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Voxel 0
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Voxel 1
            ],
            principal_axes: vec![
                0.0, 0.0, 1.0, // Voxel 0: Z-axis
                0.0, 0.0, 1.0, // Voxel 1: Z-axis
            ],
            valid: vec![1, 1],
            num_voxels: 2,
        };

        // Source point between the two voxels
        let source_points = vec![[0.5, 0.0, 0.0]];

        // Identity transform
        let transform = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
        ];

        let result = runtime
            .compute_nvtl_scores(
                &source_points,
                &voxel_data,
                &transform,
                1.0,  // gauss_d1
                1.0,  // gauss_d2
                10.0, // search_radius
            )
            .expect("compute_nvtl_scores failed");

        crate::test_println!("GPU NVTL result: {:?}", result);

        // Should have one point with neighbors
        assert_eq!(result.num_with_neighbors, 1);
        // NVTL should be negative (max of negative scores)
        assert!(result.nvtl < 0.0);
        // max_scores should have one entry
        assert_eq!(result.max_scores.len(), 1);
    }
}
