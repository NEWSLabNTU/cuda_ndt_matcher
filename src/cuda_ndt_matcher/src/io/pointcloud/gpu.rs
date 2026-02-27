//! GPU-accelerated point filtering with CPU fallback.
//!
//! This module provides sensor point filtering (distance, z-height, downsampling)
//! with automatic GPU acceleration for large point clouds (>10k points).
//! Falls back to CPU when GPU is unavailable or for small clouds.

use ndt_cuda::filtering::{CpuPointFilter, FilterParams as GpuFilterParams};

/// Parameters for filtering sensor points
#[derive(Clone, Debug)]
pub(crate) struct PointFilterParams {
    /// Minimum distance from sensor origin (default: 0.0)
    pub(crate) min_distance: f32,
    /// Maximum distance from sensor origin (default: f32::MAX)
    pub(crate) max_distance: f32,
    /// Minimum z value (ground filtering, default: -f32::MAX)
    pub(crate) min_z: f32,
    /// Maximum z value (ceiling filtering, default: f32::MAX)
    pub(crate) max_z: f32,
    /// Voxel grid downsampling resolution (None = no downsampling)
    pub(crate) downsample_resolution: Option<f32>,
}

impl Default for PointFilterParams {
    fn default() -> Self {
        Self {
            min_distance: 0.0,
            max_distance: f32::MAX,
            min_z: f32::MIN,
            max_z: f32::MAX,
            downsample_resolution: None,
        }
    }
}

/// Result of point filtering operation
#[derive(Debug)]
pub(crate) struct FilterResult {
    /// Filtered points
    pub(crate) points: Vec<[f32; 3]>,
    /// Number of points removed by distance filter
    pub(crate) removed_by_distance: usize,
    /// Number of points removed by z filter
    pub(crate) removed_by_z: usize,
    /// Number of points removed by downsampling
    pub(crate) removed_by_downsampling: usize,
    /// Whether GPU acceleration was used
    pub(crate) used_gpu: bool,
}

/// Filter sensor points based on distance and z-height constraints
///
/// This implements Autoware's sensor point preprocessing:
/// - Distance-based filtering (min/max distance from sensor origin)
/// - Z-height filtering (ground/ceiling removal)
/// - Optional voxel grid downsampling
///
/// Uses GPU acceleration for large point clouds (>10k points) when available,
/// falling back to CPU for small clouds or when GPU is unavailable.
pub(crate) fn filter_sensor_points(
    points: &[[f32; 3]],
    params: &PointFilterParams,
) -> FilterResult {
    // Convert to ndt_cuda filter params
    let gpu_params = GpuFilterParams {
        min_distance: params.min_distance,
        max_distance: params.max_distance,
        min_z: params.min_z,
        max_z: params.max_z,
        downsample_resolution: params.downsample_resolution,
    };

    // Try GPU filter first if point cloud is large enough
    if points.len() >= 10000 {
        if let Ok(gpu_filter) = ndt_cuda::GpuPointFilter::new() {
            if let Ok(result) = gpu_filter.filter(points, &gpu_params) {
                return FilterResult {
                    points: result.points,
                    removed_by_distance: result.removed_by_distance,
                    removed_by_z: result.removed_by_z,
                    removed_by_downsampling: result.removed_by_downsampling,
                    used_gpu: result.used_gpu,
                };
            }
        }
    }

    // Fall back to CPU filter
    let cpu_filter = CpuPointFilter::new();
    let result = cpu_filter.filter(points, &gpu_params);

    FilterResult {
        points: result.points,
        removed_by_distance: result.removed_by_distance,
        removed_by_z: result.removed_by_z,
        removed_by_downsampling: result.removed_by_downsampling,
        used_gpu: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_distance() {
        let points = vec![
            [1.0, 0.0, 0.0],  // distance = 1.0
            [5.0, 0.0, 0.0],  // distance = 5.0
            [10.0, 0.0, 0.0], // distance = 10.0
            [15.0, 0.0, 0.0], // distance = 15.0
        ];

        let params = PointFilterParams {
            min_distance: 3.0,
            max_distance: 12.0,
            ..Default::default()
        };

        let result = filter_sensor_points(&points, &params);
        assert_eq!(result.points.len(), 2);
        assert_eq!(result.points[0], [5.0, 0.0, 0.0]);
        assert_eq!(result.points[1], [10.0, 0.0, 0.0]);
    }

    #[test]
    fn test_filter_z_height() {
        let points = vec![
            [1.0, 0.0, -2.0], // below min_z
            [1.0, 0.0, 0.5],  // within range
            [1.0, 0.0, 1.5],  // within range
            [1.0, 0.0, 5.0],  // above max_z
        ];

        let params = PointFilterParams {
            min_z: -1.0,
            max_z: 3.0,
            ..Default::default()
        };

        let result = filter_sensor_points(&points, &params);
        assert_eq!(result.points.len(), 2);
        assert_eq!(result.removed_by_z, 2);
    }

    #[test]
    fn test_filter_with_downsampling() {
        let points = vec![
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [5.0, 0.0, 0.0], // different voxel
        ];

        let params = PointFilterParams {
            downsample_resolution: Some(1.0),
            ..Default::default()
        };

        let result = filter_sensor_points(&points, &params);
        // First two points merge into one voxel, third stays separate
        assert_eq!(result.points.len(), 2);
        assert_eq!(result.removed_by_downsampling, 1);
    }

    #[test]
    fn test_filter_default_params() {
        let points = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let params = PointFilterParams::default();

        let result = filter_sensor_points(&points, &params);
        assert_eq!(result.points.len(), 2);
        assert_eq!(result.removed_by_distance, 0);
        assert_eq!(result.removed_by_z, 0);
        assert_eq!(result.removed_by_downsampling, 0);
    }
}
