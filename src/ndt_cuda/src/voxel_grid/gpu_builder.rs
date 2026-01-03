//! GPU-accelerated voxel grid construction.
//!
//! This module provides GPU-accelerated voxel grid construction using CubeCL.
//! The approach is hybrid:
//! 1. GPU: Compute voxel IDs for all points (parallel per-point)
//! 2. CPU: Group points by voxel ID and compute statistics
//!
//! This is faster than pure CPU for large point clouds (>100K points) because
//! voxel ID computation involves 3 divisions per point which parallelize well.

use std::collections::HashMap;

use anyhow::Result;
use cubecl::client::ComputeClient;
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

use super::kernels::compute_voxel_ids_kernel;
use super::types::{Voxel, VoxelCoord, VoxelGridConfig};
use super::VoxelGrid;

/// Type alias for CUDA compute client.
type CudaClient = ComputeClient<<CudaRuntime as Runtime>::Server>;

/// GPU-accelerated voxel grid builder.
///
/// Holds a CubeCL client for GPU operations and provides methods to
/// construct voxel grids from point clouds.
pub struct GpuVoxelGridBuilder {
    client: CudaClient,
}

impl GpuVoxelGridBuilder {
    /// Create a new GPU voxel grid builder.
    ///
    /// # Returns
    /// A new builder, or error if CUDA is not available.
    pub fn new() -> Result<Self> {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::client(&device);
        Ok(Self { client })
    }

    /// Build a voxel grid from a point cloud using GPU acceleration.
    ///
    /// The GPU is used to compute voxel IDs in parallel, then CPU handles
    /// the accumulation and statistics computation.
    ///
    /// # Arguments
    /// * `points` - Input point cloud
    /// * `config` - Voxel grid configuration
    ///
    /// # Returns
    /// A VoxelGrid with computed statistics.
    pub fn build(&self, points: &[[f32; 3]], config: &VoxelGridConfig) -> Result<VoxelGrid> {
        if points.is_empty() {
            return Ok(VoxelGrid::new(config.clone()));
        }

        let num_points = points.len();

        // Step 1: Compute bounds to determine grid parameters
        let (min_bound, max_bound) = compute_point_bounds(points);
        let inv_resolution = 1.0 / config.resolution;

        // Compute grid dimensions
        let grid_dim_x = ((max_bound[0] - min_bound[0]) * inv_resolution).ceil() as u32 + 1;
        let grid_dim_y = ((max_bound[1] - min_bound[1]) * inv_resolution).ceil() as u32 + 1;

        // Step 2: Flatten points for GPU upload
        let points_flat: Vec<f32> = points.iter().flat_map(|p| p.iter().copied()).collect();

        // Step 3: Upload to GPU
        let points_gpu = self.client.create(f32::as_bytes(&points_flat));
        let min_bound_gpu = self.client.create(f32::as_bytes(&min_bound));
        let voxel_ids_gpu = self.client.empty(num_points * std::mem::size_of::<u32>());

        // Step 4: Launch kernel
        let cube_count = num_points.div_ceil(256) as u32;
        unsafe {
            compute_voxel_ids_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&points_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&min_bound_gpu, 3, 1),
                ScalarArg::new(inv_resolution),
                ScalarArg::new(grid_dim_x),
                ScalarArg::new(grid_dim_y),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<u32>(&voxel_ids_gpu, num_points, 1),
            );
        }

        // Step 5: Download voxel IDs
        let voxel_ids_bytes = self.client.read_one(voxel_ids_gpu);
        let voxel_ids = u32::from_bytes(&voxel_ids_bytes);

        // Step 6: Convert voxel IDs back to coordinates and group points
        let min_coord = VoxelCoord::new(
            (min_bound[0] * inv_resolution).floor() as i32,
            (min_bound[1] * inv_resolution).floor() as i32,
            (min_bound[2] * inv_resolution).floor() as i32,
        );

        // Group points by voxel ID
        let mut voxel_points: HashMap<u32, Vec<usize>> = HashMap::new();
        for (point_idx, &voxel_id) in voxel_ids.iter().enumerate() {
            voxel_points.entry(voxel_id).or_default().push(point_idx);
        }

        // Step 7: Compute statistics per voxel (parallel with rayon)
        let voxel_entries: Vec<_> = voxel_points.into_iter().collect();

        let voxels: Vec<_> = voxel_entries
            .into_par_iter()
            .filter_map(|(voxel_id, point_indices)| {
                if point_indices.len() < config.min_points_per_voxel {
                    return None;
                }

                // Convert voxel ID back to coordinate
                let coord = voxel_id_to_coord(voxel_id, min_coord, grid_dim_x, grid_dim_y);

                // Accumulate statistics
                let mut sum = Vector3::zeros();
                let mut sum_sq = Matrix3::zeros();

                for &idx in &point_indices {
                    let p = &points[idx];
                    let v = Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64);
                    sum += v;
                    sum_sq += v * v.transpose();
                }

                // Create voxel with statistics
                let voxel = Voxel::from_statistics(&sum, &sum_sq, point_indices.len(), config)?;
                Some((coord, voxel))
            })
            .collect();

        // Step 8: Build VoxelGrid from computed voxels
        let mut result = VoxelGrid::new(config.clone());
        for (coord, voxel) in voxels {
            result.insert(coord, voxel);
        }

        // Build search index
        result.build_search_index();

        Ok(result)
    }
}

/// Compute min/max bounds of a point cloud.
fn compute_point_bounds(points: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];

    for p in points {
        for i in 0..3 {
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }

    (min, max)
}

/// Convert voxel ID back to coordinate.
fn voxel_id_to_coord(
    id: u32,
    min_coord: VoxelCoord,
    grid_dim_x: u32,
    grid_dim_y: u32,
) -> VoxelCoord {
    let dim_xy = grid_dim_x * grid_dim_y;
    let z = id / dim_xy;
    let remainder = id % dim_xy;
    let y = remainder / grid_dim_x;
    let x = remainder % grid_dim_x;

    VoxelCoord::new(
        x as i32 + min_coord.x,
        y as i32 + min_coord.y,
        z as i32 + min_coord.z,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::make_default_half_cubic_pcd;

    #[test]
    fn test_gpu_voxel_grid_construction() {
        let points = make_default_half_cubic_pcd();
        let config = VoxelGridConfig::default();

        let builder = GpuVoxelGridBuilder::new().expect("Failed to create GPU builder");
        let grid = builder
            .build(&points, &config)
            .expect("Failed to build grid");

        // Should produce voxels
        assert!(!grid.is_empty());
        assert!(grid.len() > 100, "Expected >100 voxels for half-cubic PCD");
    }

    #[test]
    fn test_gpu_cpu_consistency() {
        let points = make_default_half_cubic_pcd();
        let config = VoxelGridConfig::default();

        // Build with GPU
        let builder = GpuVoxelGridBuilder::new().expect("Failed to create GPU builder");
        let gpu_grid = builder
            .build(&points, &config)
            .expect("Failed to build GPU grid");

        // Build with CPU
        let cpu_grid = VoxelGrid::from_points_with_config(&points, config.clone())
            .expect("Failed to build CPU grid");

        // Should have similar voxel counts
        let diff = (gpu_grid.len() as i32 - cpu_grid.len() as i32).abs();
        assert!(
            diff <= cpu_grid.len() as i32 / 10,
            "GPU ({}) and CPU ({}) voxel counts should be similar",
            gpu_grid.len(),
            cpu_grid.len()
        );
    }
}
