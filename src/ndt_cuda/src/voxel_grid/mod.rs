//! Voxel grid construction and management.
//!
//! This module provides GPU-accelerated voxel grid construction from point clouds.
//! The voxel grid stores Gaussian distributions (mean + covariance) for each voxel,
//! which are used for NDT scan matching.
//!
//! # Architecture
//!
//! The implementation uses a hybrid CPU/GPU approach:
//! 1. GPU: Compute voxel IDs for all points (parallel)
//! 2. CPU: Accumulate point statistics per voxel (HashMap-based)
//! 3. CPU: Compute covariance and regularization (parallel via rayon)
//! 4. CPU: Build KD-tree from voxel centroids for radius search
//!
//! The KD-tree enables efficient radius search matching Autoware's behavior,
//! where each source point can contribute to score from multiple nearby voxels.

pub mod cpu;
pub mod gpu;
pub mod kernels;
pub mod search;
pub mod types;

pub use search::VoxelSearch;
pub use types::{Voxel, VoxelCoord, VoxelGridConfig};

use std::collections::HashMap;

use anyhow::Result;

use crate::voxel_grid::cpu::build_voxel_grid_cpu;

/// A voxel grid containing Gaussian distributions for NDT matching.
///
/// The grid stores:
/// - Voxel means (centroids)
/// - Inverse covariance matrices
/// - Spatial hash for O(1) voxel lookup
/// - KD-tree for efficient radius search (like Autoware's radiusSearch)
#[derive(Debug)]
pub struct VoxelGrid {
    /// Configuration used to build this grid.
    pub config: VoxelGridConfig,
    /// Voxels stored in a vector for indexed access.
    voxels: Vec<Voxel>,
    /// Voxel coordinates (parallel to voxels vector).
    coords: Vec<VoxelCoord>,
    /// Map from coordinate to index in voxels vector.
    coord_to_index: HashMap<VoxelCoord, usize>,
    /// KD-tree for radius search over voxel centroids.
    search: Option<VoxelSearch>,
    /// Precomputed bounds for GPU operations.
    min_bound: Option<VoxelCoord>,
    max_bound: Option<VoxelCoord>,
    grid_dims: Option<[u32; 3]>,
}

impl VoxelGrid {
    /// Create a new empty voxel grid with the given configuration.
    pub fn new(config: VoxelGridConfig) -> Self {
        Self {
            config,
            voxels: Vec::new(),
            coords: Vec::new(),
            coord_to_index: HashMap::new(),
            search: None,
            min_bound: None,
            max_bound: None,
            grid_dims: None,
        }
    }

    /// Build a voxel grid from a point cloud.
    ///
    /// This is the main entry point for voxel grid construction.
    /// Currently uses CPU implementation; GPU acceleration coming in future.
    ///
    /// # Arguments
    /// * `points` - Input point cloud (array of [x, y, z] coordinates)
    /// * `resolution` - Voxel side length in meters
    ///
    /// # Example
    /// ```ignore
    /// let points = vec![[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], /* ... */];
    /// let grid = VoxelGrid::from_points(&points, 2.0)?;
    /// ```
    pub fn from_points(points: &[[f32; 3]], resolution: f32) -> Result<Self> {
        let config = VoxelGridConfig {
            resolution,
            ..Default::default()
        };
        Self::from_points_with_config(points, config)
    }

    /// Build a voxel grid with custom configuration.
    pub fn from_points_with_config(points: &[[f32; 3]], config: VoxelGridConfig) -> Result<Self> {
        let voxel_map = build_voxel_grid_cpu(points, &config);

        // Convert HashMap to Vec + index map
        let mut voxels = Vec::with_capacity(voxel_map.len());
        let mut coords = Vec::with_capacity(voxel_map.len());
        let mut coord_to_index = HashMap::with_capacity(voxel_map.len());

        for (coord, voxel) in voxel_map {
            let idx = voxels.len();
            coords.push(coord);
            voxels.push(voxel);
            coord_to_index.insert(coord, idx);
        }

        // Compute bounds
        let (min_bound, max_bound, grid_dims) = if coords.is_empty() {
            (None, None, None)
        } else {
            let (min, max, dims) = cpu::compute_voxel_bounds(&coords);
            (Some(min), Some(max), Some(dims))
        };

        // Build KD-tree for radius search
        let search = VoxelSearch::from_voxels(&voxels);

        Ok(Self {
            config,
            voxels,
            coords,
            coord_to_index,
            search,
            min_bound,
            max_bound,
            grid_dims,
        })
    }

    /// Get the number of voxels in the grid.
    pub fn len(&self) -> usize {
        self.voxels.len()
    }

    /// Check if the grid is empty.
    pub fn is_empty(&self) -> bool {
        self.voxels.is_empty()
    }

    /// Get a voxel by its coordinates.
    pub fn get(&self, coord: &VoxelCoord) -> Option<&Voxel> {
        self.coord_to_index.get(coord).map(|&idx| &self.voxels[idx])
    }

    /// Get a voxel by integer coordinate array [x, y, z].
    pub fn get_by_coord(&self, coord: &[i32; 3]) -> Option<&Voxel> {
        let voxel_coord = VoxelCoord {
            x: coord[0],
            y: coord[1],
            z: coord[2],
        };
        self.get(&voxel_coord)
    }

    /// Get a voxel by point position (geometric containment).
    ///
    /// Returns the single voxel that geometrically contains this point.
    /// For multi-voxel queries, use `radius_search` instead.
    pub fn get_by_point(&self, point: &[f32; 3]) -> Option<&Voxel> {
        let coord = VoxelCoord::from_point(point, self.config.resolution);
        self.get(&coord)
    }

    /// Get a voxel by its index.
    pub fn get_by_index(&self, idx: usize) -> Option<&Voxel> {
        self.voxels.get(idx)
    }

    /// Find all voxels within a given radius of a point.
    ///
    /// This matches Autoware's `radiusSearch` behavior where each source point
    /// may contribute to score from multiple voxels, providing smoother gradients
    /// especially near voxel boundaries.
    ///
    /// Uses KD-tree for efficient search over voxel centroids (means).
    ///
    /// # Arguments
    /// * `point` - Query point [x, y, z]
    /// * `radius` - Search radius (typically equal to voxel resolution)
    ///
    /// # Returns
    /// Vector of references to voxels within the radius, sorted by distance.
    pub fn radius_search(&self, point: &[f32; 3], radius: f32) -> Vec<&Voxel> {
        match &self.search {
            Some(search) => {
                let indices = search.within(point, radius);
                indices.iter().map(|&idx| &self.voxels[idx]).collect()
            }
            None => Vec::new(),
        }
    }

    /// Find all voxel indices within a given radius of a point.
    ///
    /// Returns indices that can be used with `get_by_index`.
    pub fn radius_search_indices(&self, point: &[f32; 3], radius: f32) -> Vec<usize> {
        match &self.search {
            Some(search) => search.within(point, radius),
            None => Vec::new(),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &VoxelGridConfig {
        &self.config
    }

    /// Iterate over all voxels with their coordinates.
    pub fn iter(&self) -> impl Iterator<Item = (&VoxelCoord, &Voxel)> {
        self.coords.iter().zip(self.voxels.iter())
    }

    /// Iterate over all voxels.
    pub fn voxels(&self) -> &[Voxel] {
        &self.voxels
    }

    /// Get the grid bounds.
    pub fn bounds(&self) -> Option<(VoxelCoord, VoxelCoord)> {
        match (self.min_bound, self.max_bound) {
            (Some(min), Some(max)) => Some((min, max)),
            _ => None,
        }
    }

    /// Get the grid dimensions.
    pub fn dims(&self) -> Option<[u32; 3]> {
        self.grid_dims
    }

    /// Get the resolution.
    pub fn resolution(&self) -> f32 {
        self.config.resolution
    }

    /// Get all voxel means as a flat array [V * 3].
    ///
    /// Useful for GPU upload.
    pub fn means_flat(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.voxels.len() * 3);
        for voxel in &self.voxels {
            result.push(voxel.mean.x);
            result.push(voxel.mean.y);
            result.push(voxel.mean.z);
        }
        result
    }

    /// Get all inverse covariances as a flat array [V * 9].
    ///
    /// Row-major 3x3 matrices. Useful for GPU upload.
    pub fn inv_covariances_flat(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.voxels.len() * 9);
        for voxel in &self.voxels {
            // Row-major order
            for row in 0..3 {
                for col in 0..3 {
                    result.push(voxel.inv_covariance[(row, col)]);
                }
            }
        }
        result
    }

    /// Get all voxel coordinates as a flat array.
    pub fn coords_flat(&self) -> Vec<i32> {
        let mut result = Vec::with_capacity(self.coords.len() * 3);
        for coord in &self.coords {
            result.push(coord.x);
            result.push(coord.y);
            result.push(coord.z);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn generate_test_points() -> Vec<[f32; 3]> {
        use rand::prelude::*;
        use rand_distr::Normal;

        let mut rng = rand::thread_rng();
        // Use very small spread to ensure points stay within one voxel (resolution 2.0)
        let dist = Normal::new(0.0, 0.1).unwrap();

        // Choose centers well inside voxel boundaries
        // With resolution 2.0, voxel [0,0,0] covers [-inf,2) in each dim
        // voxel [5,0,0] covers [10,12) in x
        // voxel [0,5,0] covers [10,12) in y
        let centers = [[1.0, 1.0, 1.0], [11.0, 1.0, 1.0], [1.0, 11.0, 1.0]];
        let mut points = Vec::new();

        for center in &centers {
            for _ in 0..50 {
                points.push([
                    center[0] + dist.sample(&mut rng) as f32,
                    center[1] + dist.sample(&mut rng) as f32,
                    center[2] + dist.sample(&mut rng) as f32,
                ]);
            }
        }

        points
    }

    #[test]
    fn test_voxel_grid_from_points() {
        let points = generate_test_points();
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        // Should have 3 voxels (one per cluster)
        // Each cluster has 50 points, well above the min_points_per_voxel threshold
        assert_eq!(grid.len(), 3, "Expected 3 voxels but got {}", grid.len());

        // Check resolution
        assert_eq!(grid.resolution(), 2.0);
    }

    #[test]
    fn test_voxel_grid_get_by_point() {
        let points = generate_test_points();
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        // Query a point in the first cluster
        let voxel = grid.get_by_point(&[0.1, 0.1, 0.1]);
        assert!(voxel.is_some());

        let voxel = voxel.unwrap();
        assert!(voxel.point_count >= 6);
    }

    #[test]
    fn test_voxel_grid_means_flat() {
        let points = generate_test_points();
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        let means = grid.means_flat();
        assert_eq!(means.len(), grid.len() * 3);
    }

    #[test]
    fn test_voxel_grid_inv_covariances_flat() {
        let points = generate_test_points();
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        let inv_covs = grid.inv_covariances_flat();
        assert_eq!(inv_covs.len(), grid.len() * 9);

        // Verify no NaN or Inf values
        for val in &inv_covs {
            assert!(val.is_finite(), "Found non-finite value: {val}");
        }
    }

    #[test]
    fn test_voxel_grid_bounds() {
        let points = generate_test_points();
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        let bounds = grid.bounds();
        assert!(bounds.is_some());

        let (min, max) = bounds.unwrap();
        // First cluster at origin should give voxel (0,0,0)
        // Second cluster at (10,0,0) should give voxel (5,0,0)
        // Third cluster at (0,10,0) should give voxel (0,5,0)
        assert!(min.x <= 0);
        assert!(min.y <= 0);
        assert!(max.x >= 4); // 10/2 = 5, but floor
        assert!(max.y >= 4);
    }

    #[test]
    fn test_empty_voxel_grid() {
        let points: Vec<[f32; 3]> = Vec::new();
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        assert!(grid.is_empty());
        assert_eq!(grid.len(), 0);
        assert!(grid.bounds().is_none());
    }

    #[test]
    fn test_voxel_mean_accuracy() {
        // Create points with known mean
        // Choose a center well inside a voxel boundary with resolution 2.0
        // Center at 5.0 -> voxel coord 2, voxel covers [4, 6)
        // Points at 5.0 +/- 0.1 stay within [4.9, 5.1] which is inside [4, 6)
        let center = [5.0f32, 5.0, 5.0];
        let mut points = Vec::new();

        // Generate points in a tight cluster around the center
        for dx in [-0.1, 0.0, 0.1] {
            for dy in [-0.1, 0.0, 0.1] {
                for dz in [-0.1, 0.0, 0.1] {
                    points.push([
                        center[0] + dx as f32,
                        center[1] + dy as f32,
                        center[2] + dz as f32,
                    ]);
                }
            }
        }

        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();
        assert_eq!(grid.len(), 1, "Expected 1 voxel but got {}", grid.len());

        let coord = VoxelCoord::from_point(&center, 2.0);
        let voxel = grid.get(&coord).unwrap();

        // Mean should be very close to center
        assert_relative_eq!(voxel.mean.x, center[0], epsilon = 0.02);
        assert_relative_eq!(voxel.mean.y, center[1], epsilon = 0.02);
        assert_relative_eq!(voxel.mean.z, center[2], epsilon = 0.02);
    }

    #[test]
    fn test_covariance_symmetry() {
        let points = generate_test_points();
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        for (_, voxel) in grid.iter() {
            let cov = &voxel.covariance;
            let inv_cov = &voxel.inv_covariance;

            // Check symmetry
            for i in 0..3 {
                for j in 0..3 {
                    assert_relative_eq!(cov[(i, j)], cov[(j, i)], epsilon = 1e-6);
                    assert_relative_eq!(inv_cov[(i, j)], inv_cov[(j, i)], epsilon = 1e-6);
                }
            }

            // Check that cov * inv_cov ≈ I
            let product = cov * inv_cov;
            for i in 0..3 {
                for j in 0..3 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert_relative_eq!(product[(i, j)], expected, epsilon = 0.01);
                }
            }
        }
    }
}
