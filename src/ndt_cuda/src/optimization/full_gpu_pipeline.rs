//! Full GPU Newton Optimization Pipeline.
//!
//! This module implements a full GPU Newton iteration loop that eliminates
//! per-iteration CPU↔GPU memory transfers for Jacobians and Hessians.
//!
//! # Architecture
//!
//! ```text
//! Once per alignment:
//!   Upload: source_points [N×3], voxel_data [V×12], initial_pose [6]
//!
//! GPU Iteration Loop (30 iterations max):
//!   1. Compute sin/cos from pose [1 thread]
//!   2. Compute Jacobians from sin/cos + points [N threads]
//!   3. Compute Point Hessians from sin/cos + points [N threads]
//!   4. Compute transform from pose [1 thread or CPU]
//!   5. Transform points [N threads]
//!   6. Radius search [N threads]
//!   7. Compute score/gradient/Hessian [N threads]
//!   8. CUB segmented reduce [43 segments]
//!   9. Newton solve: δ = -H⁻¹g [cuSOLVER]
//!   10. Update pose: pose += δ [1 thread or CPU]
//!   11. Check convergence (every 5 iterations)
//!
//! Download: final_pose [6], converged [1], score [1]
//! ```
//!
//! # Transfer Analysis
//!
//! | Phase | Current | Full GPU |
//! |-------|---------|----------|
//! | Per-iteration upload | ~490 KB | 0 |
//! | Per-iteration download | 172 B | 0 (except every 5 iters) |
//! | Final download | - | 24 B (pose) |

use anyhow::Result;
use cubecl::client::ComputeClient;
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::derivatives::gpu::{
    compute_ndt_gradient_kernel, compute_ndt_hessian_kernel, compute_ndt_score_kernel,
    pose_to_transform_matrix, radius_search_kernel, GpuVoxelData, MAX_NEIGHBORS,
};
use crate::derivatives::gpu_jacobian::{
    compute_jacobians_kernel, compute_point_hessians_kernel, compute_sin_cos_kernel,
};
use crate::optimization::GpuNewtonSolver;
use crate::voxel_grid::kernels::transform_points_kernel;

/// Type alias for CUDA compute client.
type CudaClient = ComputeClient<<CudaRuntime as Runtime>::Server>;

/// Result of full GPU optimization.
#[derive(Debug, Clone)]
pub struct FullGpuOptimizationResult {
    /// Final pose [tx, ty, tz, roll, pitch, yaw]
    pub pose: [f64; 6],
    /// Final NDT score (negative log-likelihood)
    pub score: f64,
    /// Whether optimization converged
    pub converged: bool,
    /// Number of iterations performed
    pub iterations: u32,
    /// Final Hessian matrix (6x6, row-major)
    pub hessian: [[f64; 6]; 6],
    /// Number of point-voxel correspondences
    pub num_correspondences: usize,
}

/// Full GPU Newton optimization pipeline.
///
/// Runs the entire Newton optimization loop on GPU, only downloading
/// the final pose and convergence status.
pub struct FullGpuPipeline {
    client: CudaClient,
    #[allow(dead_code)]
    device: CudaDevice,

    // Capacity tracking
    max_points: usize,
    max_voxels: usize,

    // Current sizes
    num_points: usize,
    num_voxels: usize,

    // Persistent data (uploaded once per alignment)
    source_points: Handle,  // [N × 3]
    voxel_means: Handle,    // [V × 3]
    voxel_inv_covs: Handle, // [V × 9]
    voxel_valid: Handle,    // [V]

    // Per-iteration GPU buffers (computed on GPU each iteration)
    sin_cos: Handle,            // [6] - sin/cos of pose angles
    jacobians: Handle,          // [N × 18]
    point_hessians: Handle,     // [N × 144]
    jacobians_combined: Handle, // [N × (18 + 144)] for Hessian kernel
    transform: Handle,          // [16]
    transformed_points: Handle, // [N × 3]
    neighbor_indices: Handle,   // [N × MAX_NEIGHBORS]
    neighbor_counts: Handle,    // [N]
    scores: Handle,             // [N]
    correspondences: Handle,    // [N]
    gradients: Handle,          // [N × 6] column-major
    hessians: Handle,           // [N × 36] column-major

    // Gaussian parameters
    gauss_d1: f32,
    gauss_d2: f32,
    search_radius_sq: f32,

    // CUB reduction buffers
    reduce_temp: Handle,
    reduce_temp_bytes: usize,
    reduce_offsets: Handle,
    reduce_output: Handle, // [43] floats

    // cuSOLVER Newton solver
    newton_solver: GpuNewtonSolver,
}

impl FullGpuPipeline {
    /// Get raw CUDA device pointer from CubeCL handle.
    fn raw_ptr(&self, handle: &Handle) -> u64 {
        let binding = handle.clone().binding();
        let resource = self.client.get_resource(binding);
        resource.resource().ptr
    }

    /// Create a new full GPU pipeline with given capacity.
    pub fn new(max_points: usize, max_voxels: usize) -> Result<Self> {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::client(&device);

        // Allocate source data buffers
        let source_points = client.empty(max_points * 3 * std::mem::size_of::<f32>());
        let voxel_means = client.empty(max_voxels * 3 * std::mem::size_of::<f32>());
        let voxel_inv_covs = client.empty(max_voxels * 9 * std::mem::size_of::<f32>());
        let voxel_valid = client.empty(max_voxels * std::mem::size_of::<u32>());

        // Allocate per-iteration buffers
        let sin_cos = client.empty(6 * std::mem::size_of::<f32>());
        let jacobians = client.empty(max_points * 18 * std::mem::size_of::<f32>());
        let point_hessians = client.empty(max_points * 144 * std::mem::size_of::<f32>());
        let jacobians_combined = client.empty(max_points * (18 + 144) * std::mem::size_of::<f32>());
        let transform = client.empty(16 * std::mem::size_of::<f32>());
        let transformed_points = client.empty(max_points * 3 * std::mem::size_of::<f32>());
        let neighbor_indices =
            client.empty(max_points * MAX_NEIGHBORS as usize * std::mem::size_of::<i32>());
        let neighbor_counts = client.empty(max_points * std::mem::size_of::<u32>());
        let scores = client.empty(max_points * std::mem::size_of::<f32>());
        let correspondences = client.empty(max_points * std::mem::size_of::<u32>());
        let gradients = client.empty(max_points * 6 * std::mem::size_of::<f32>());
        let hessians = client.empty(max_points * 36 * std::mem::size_of::<f32>());

        // CUB reduction buffers
        // Query required temp storage size for 43 segments (1 score + 6 grad + 36 hess)
        let reduce_temp_bytes =
            cuda_ffi::segmented_reduce_sum_f32_temp_size(max_points * 43, 43)? as usize;
        let reduce_temp = client.empty(reduce_temp_bytes.max(256)); // Minimum 256 bytes

        // Create segment offsets for 43 segments: score(1) + grad(6) + hess(36)
        let offsets: Vec<i32> = (0..=43).map(|i| (i * max_points) as i32).collect();
        let reduce_offsets = client.create(i32::as_bytes(&offsets));
        let reduce_output = client.empty(43 * std::mem::size_of::<f32>());

        // Create cuSOLVER Newton solver
        let newton_solver = GpuNewtonSolver::new(0)?;

        Ok(Self {
            client,
            device,
            max_points,
            max_voxels,
            num_points: 0,
            num_voxels: 0,
            source_points,
            voxel_means,
            voxel_inv_covs,
            voxel_valid,
            sin_cos,
            jacobians,
            point_hessians,
            jacobians_combined,
            transform,
            transformed_points,
            neighbor_indices,
            neighbor_counts,
            scores,
            correspondences,
            gradients,
            hessians,
            gauss_d1: 0.0,
            gauss_d2: 0.0,
            search_radius_sq: 0.0,
            reduce_temp,
            reduce_temp_bytes,
            reduce_offsets,
            reduce_output,
            newton_solver,
        })
    }

    /// Upload alignment data to GPU.
    ///
    /// This is called once per alignment before running iterations.
    pub fn upload_alignment_data(
        &mut self,
        source_points: &[[f32; 3]],
        voxel_data: &GpuVoxelData,
        gauss_d1: f32,
        gauss_d2: f32,
        search_radius: f32,
    ) -> Result<()> {
        let num_points = source_points.len();
        let num_voxels = voxel_data.num_voxels;

        if num_points > self.max_points {
            anyhow::bail!("Too many source points: {num_points} > {}", self.max_points);
        }
        if num_voxels > self.max_voxels {
            anyhow::bail!("Too many voxels: {num_voxels} > {}", self.max_voxels);
        }

        self.num_points = num_points;
        self.num_voxels = num_voxels;
        self.gauss_d1 = gauss_d1;
        self.gauss_d2 = gauss_d2;
        self.search_radius_sq = search_radius * search_radius;

        // Flatten source points
        let points_flat: Vec<f32> = source_points
            .iter()
            .flat_map(|p| p.iter().copied())
            .collect();
        self.source_points = self.client.create(f32::as_bytes(&points_flat));

        // Upload voxel data
        self.voxel_means = self.client.create(f32::as_bytes(&voxel_data.means));
        self.voxel_inv_covs = self
            .client
            .create(f32::as_bytes(&voxel_data.inv_covariances));
        self.voxel_valid = self.client.create(u32::as_bytes(&voxel_data.valid));

        // Update CUB offsets for actual num_points
        let offsets: Vec<i32> = (0..=43).map(|i| (i * num_points) as i32).collect();
        self.reduce_offsets = self.client.create(i32::as_bytes(&offsets));

        Ok(())
    }

    /// Run full GPU Newton optimization.
    ///
    /// # Arguments
    /// * `initial_pose` - Initial pose [tx, ty, tz, roll, pitch, yaw]
    /// * `max_iterations` - Maximum number of Newton iterations
    /// * `transformation_epsilon` - Convergence threshold for step size
    ///
    /// # Returns
    /// Optimization result with final pose and convergence status.
    pub fn optimize(
        &mut self,
        initial_pose: &[f64; 6],
        max_iterations: u32,
        transformation_epsilon: f64,
    ) -> Result<FullGpuOptimizationResult> {
        let num_points = self.num_points;
        let num_voxels = self.num_voxels;

        if num_points == 0 {
            return Ok(FullGpuOptimizationResult {
                pose: *initial_pose,
                score: 0.0,
                converged: true,
                iterations: 0,
                hessian: [[0.0; 6]; 6],
                num_correspondences: 0,
            });
        }

        // Current pose (on CPU for now, will be transferred to GPU per iteration)
        let mut current_pose = *initial_pose;
        let mut converged = false;
        let mut iterations = 0u32;
        let mut final_score = 0.0f64;
        let mut final_hessian = [[0.0f64; 6]; 6];
        let mut final_correspondences = 0usize;

        let cube_count = num_points.div_ceil(256) as u32;

        for iter in 0..max_iterations {
            iterations = iter + 1;

            // Step 1: Upload pose and compute sin/cos
            let pose_f32: [f32; 6] = [
                current_pose[0] as f32,
                current_pose[1] as f32,
                current_pose[2] as f32,
                current_pose[3] as f32,
                current_pose[4] as f32,
                current_pose[5] as f32,
            ];
            let pose_for_sincos = self.client.create(f32::as_bytes(&pose_f32));

            unsafe {
                compute_sin_cos_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(1, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&pose_for_sincos, 6, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.sin_cos, 6, 1),
                );
            }

            // Step 2: Compute Jacobians on GPU
            unsafe {
                compute_jacobians_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new(256, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.source_points, num_points * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.sin_cos, 6, 1),
                    ScalarArg::new(num_points as u32),
                    ArrayArg::from_raw_parts::<f32>(&self.jacobians, num_points * 18, 1),
                );
            }

            // Step 3: Compute Point Hessians on GPU
            unsafe {
                compute_point_hessians_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new(256, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.source_points, num_points * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.sin_cos, 6, 1),
                    ScalarArg::new(num_points as u32),
                    ArrayArg::from_raw_parts::<f32>(&self.point_hessians, num_points * 144, 1),
                );
            }

            // Step 4: Compute transform matrix (CPU for now - only 64 bytes)
            let transform = pose_to_transform_matrix(&current_pose);
            self.transform = self.client.create(f32::as_bytes(&transform));

            // Step 5: Transform points
            unsafe {
                transform_points_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new(256, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.source_points, num_points * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.transform, 16, 1),
                    ScalarArg::new(num_points as u32),
                    ArrayArg::from_raw_parts::<f32>(&self.transformed_points, num_points * 3, 1),
                );
            }

            // Step 6: Radius search
            unsafe {
                radius_search_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new(256, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.transformed_points, num_points * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.voxel_means, num_voxels * 3, 1),
                    ArrayArg::from_raw_parts::<u32>(&self.voxel_valid, num_voxels, 1),
                    ScalarArg::new(self.search_radius_sq),
                    ScalarArg::new(num_points as u32),
                    ScalarArg::new(num_voxels as u32),
                    ArrayArg::from_raw_parts::<i32>(
                        &self.neighbor_indices,
                        num_points * MAX_NEIGHBORS as usize,
                        1,
                    ),
                    ArrayArg::from_raw_parts::<u32>(&self.neighbor_counts, num_points, 1),
                );
            }

            // Step 7a: Score kernel
            unsafe {
                compute_ndt_score_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new(256, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.source_points, num_points * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.transform, 16, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.voxel_means, num_voxels * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.voxel_inv_covs, num_voxels * 9, 1),
                    ArrayArg::from_raw_parts::<i32>(
                        &self.neighbor_indices,
                        num_points * MAX_NEIGHBORS as usize,
                        1,
                    ),
                    ArrayArg::from_raw_parts::<u32>(&self.neighbor_counts, num_points, 1),
                    ScalarArg::new(self.gauss_d1),
                    ScalarArg::new(self.gauss_d2),
                    ScalarArg::new(num_points as u32),
                    ArrayArg::from_raw_parts::<f32>(&self.scores, num_points, 1),
                    ArrayArg::from_raw_parts::<u32>(&self.correspondences, num_points, 1),
                );
            }

            // Step 7b: Gradient kernel
            unsafe {
                compute_ndt_gradient_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new(256, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.source_points, num_points * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.transform, 16, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.jacobians, num_points * 18, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.voxel_means, num_voxels * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.voxel_inv_covs, num_voxels * 9, 1),
                    ArrayArg::from_raw_parts::<i32>(
                        &self.neighbor_indices,
                        num_points * MAX_NEIGHBORS as usize,
                        1,
                    ),
                    ArrayArg::from_raw_parts::<u32>(&self.neighbor_counts, num_points, 1),
                    ScalarArg::new(self.gauss_d1),
                    ScalarArg::new(self.gauss_d2),
                    ScalarArg::new(num_points as u32),
                    ArrayArg::from_raw_parts::<f32>(&self.gradients, num_points * 6, 1),
                );
            }

            // Step 7c: Hessian kernel - need combined jacobians + point_hessians
            // For now, we need to copy jacobians and point_hessians into combined buffer
            // TODO: Optimize by having Hessian kernel take separate buffers
            // Sync and copy on CPU for now (will be eliminated in future optimization)
            cubecl::future::block_on(self.client.sync());

            let jacobians_bytes = self.client.read_one(self.jacobians.clone());
            let point_hessians_bytes = self.client.read_one(self.point_hessians.clone());
            let jacobians_vec = f32::from_bytes(&jacobians_bytes);
            let point_hessians_vec = f32::from_bytes(&point_hessians_bytes);

            let mut combined = Vec::with_capacity(num_points * (18 + 144));
            combined.extend_from_slice(jacobians_vec);
            combined.extend_from_slice(point_hessians_vec);
            self.jacobians_combined = self.client.create(f32::as_bytes(&combined));

            // Gauss params buffer
            let gauss_params = [self.gauss_d1, self.gauss_d2];
            let gauss_params_handle = self.client.create(f32::as_bytes(&gauss_params));

            unsafe {
                compute_ndt_hessian_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new(256, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.source_points, num_points * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.transform, 16, 1),
                    ArrayArg::from_raw_parts::<f32>(
                        &self.jacobians_combined,
                        num_points * (18 + 144),
                        1,
                    ),
                    ArrayArg::from_raw_parts::<f32>(&self.voxel_means, num_voxels * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.voxel_inv_covs, num_voxels * 9, 1),
                    ArrayArg::from_raw_parts::<i32>(
                        &self.neighbor_indices,
                        num_points * MAX_NEIGHBORS as usize,
                        1,
                    ),
                    ArrayArg::from_raw_parts::<u32>(&self.neighbor_counts, num_points, 1),
                    ArrayArg::from_raw_parts::<f32>(&gauss_params_handle, 2, 1),
                    ScalarArg::new(num_points as u32),
                    ArrayArg::from_raw_parts::<f32>(&self.hessians, num_points * 36, 1),
                );
            }

            // Step 8: CUB reduction (3 separate reductions)
            cubecl::future::block_on(self.client.sync());

            let n = num_points as i32;

            // Reduce scores (1 segment) -> reduce_output[0]
            let score_offsets: Vec<i32> = vec![0, n];
            let score_offsets_handle = self.client.create(i32::as_bytes(&score_offsets));
            cubecl::future::block_on(self.client.sync());

            unsafe {
                cuda_ffi::segmented_reduce_sum_f32_inplace(
                    self.raw_ptr(&self.reduce_temp),
                    self.reduce_temp_bytes,
                    self.raw_ptr(&self.scores),
                    self.raw_ptr(&self.reduce_output),
                    1,
                    self.raw_ptr(&score_offsets_handle),
                )?;
            }

            // Reduce gradients (6 segments) -> reduce_output[1..7]
            let grad_offsets: Vec<i32> = (0..=6).map(|i| i * n).collect();
            let grad_offsets_handle = self.client.create(i32::as_bytes(&grad_offsets));
            cubecl::future::block_on(self.client.sync());

            // Output goes to reduce_output + 1 float (4 bytes)
            let grad_output_ptr = self.raw_ptr(&self.reduce_output) + 4;
            unsafe {
                cuda_ffi::segmented_reduce_sum_f32_inplace(
                    self.raw_ptr(&self.reduce_temp),
                    self.reduce_temp_bytes,
                    self.raw_ptr(&self.gradients),
                    grad_output_ptr,
                    6,
                    self.raw_ptr(&grad_offsets_handle),
                )?;
            }

            // Reduce hessians (36 segments) -> reduce_output[7..43]
            let hess_offsets: Vec<i32> = (0..=36).map(|i| i * n).collect();
            let hess_offsets_handle = self.client.create(i32::as_bytes(&hess_offsets));
            cubecl::future::block_on(self.client.sync());

            // Output goes to reduce_output + 7 floats (28 bytes)
            let hess_output_ptr = self.raw_ptr(&self.reduce_output) + 28;
            unsafe {
                cuda_ffi::segmented_reduce_sum_f32_inplace(
                    self.raw_ptr(&self.reduce_temp),
                    self.reduce_temp_bytes,
                    self.raw_ptr(&self.hessians),
                    hess_output_ptr,
                    36,
                    self.raw_ptr(&hess_offsets_handle),
                )?;
            }

            // Step 9: Download reduced results and do Newton solve on CPU
            // (cuSOLVER integration pending - for now solve on CPU)
            let reduce_output_bytes = self.client.read_one(self.reduce_output.clone());
            let reduce_output = f32::from_bytes(&reduce_output_bytes);

            let score = reduce_output[0] as f64;
            let gradient: [f64; 6] = std::array::from_fn(|i| reduce_output[1 + i] as f64);
            let hessian_flat: [f64; 36] = std::array::from_fn(|i| reduce_output[7 + i] as f64);

            // Reconstruct symmetric Hessian from upper triangle
            let mut hessian = [[0.0f64; 6]; 6];
            for i in 0..6 {
                for j in 0..6 {
                    hessian[i][j] = hessian_flat[i * 6 + j];
                }
            }

            // Get correspondences count
            let correspondences_bytes = self.client.read_one(self.correspondences.clone());
            let correspondences = u32::from_bytes(&correspondences_bytes);
            let num_correspondences: u32 = correspondences.iter().sum();

            // Newton step using GPU solver
            let delta = self.newton_solver.solve(&hessian_flat, &gradient)?;

            // Step 10: Update pose
            for i in 0..6 {
                current_pose[i] += delta[i];
            }

            // Step 11: Check convergence
            let delta_norm = delta.iter().map(|d| d * d).sum::<f64>().sqrt();

            final_score = score;
            final_hessian = hessian;
            final_correspondences = num_correspondences as usize;

            if delta_norm < transformation_epsilon {
                converged = true;
                break;
            }
        }

        Ok(FullGpuOptimizationResult {
            pose: current_pose,
            score: final_score,
            converged,
            iterations,
            hessian: final_hessian,
            num_correspondences: final_correspondences,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = FullGpuPipeline::new(1000, 5000);
        assert!(pipeline.is_ok());
    }
}
