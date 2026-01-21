//! Visualization utilities for NDT debugging.
//!
//! This module provides functions for visualizing NDT data, including:
//! - Per-point score visualization (colored point clouds)
//! - Pose history markers
//! - Particle filter visualization (multiple color schemes)

use crate::particle::Particle;
use builtin_interfaces::msg::Duration as RosDuration;
use geometry_msgs::msg::Vector3;
use ndt_cuda::scoring::colors::{
    color_to_rgb_packed, ndt_score_to_color, DEFAULT_SCORE_LOWER, DEFAULT_SCORE_UPPER,
};
use ndt_cuda::scoring::nvtl::{compute_nvtl, NvtlConfig};
use ndt_cuda::GaussianParams;
use ndt_cuda::VoxelGrid;
use sensor_msgs::msg::PointCloud2;
use sensor_msgs::msg::PointField;
use std_msgs::msg::ColorRGBA;
use std_msgs::msg::Header;
use visualization_msgs::msg::{Marker, MarkerArray};

/// Configuration for point score visualization.
#[derive(Debug, Clone)]
pub struct PointScoreConfig {
    /// Lower bound for score color mapping.
    pub score_lower: f32,
    /// Upper bound for score color mapping.
    pub score_upper: f32,
    /// Search radius for NVTL computation (in voxel units).
    pub search_radius: i32,
}

impl Default for PointScoreConfig {
    fn default() -> Self {
        Self {
            score_lower: DEFAULT_SCORE_LOWER,
            score_upper: DEFAULT_SCORE_UPPER,
            search_radius: 1,
        }
    }
}

/// Generate a colored point cloud showing per-point NDT scores.
///
/// For each source point:
/// 1. Transform by the given pose
/// 2. Compute nearest voxel score (max score across neighbors)
/// 3. Map score to RGB color
/// 4. Output as PointXYZRGB
///
/// This matches Autoware's `visualize_point_score` function.
///
/// # Arguments
/// * `source_points` - Source point cloud
/// * `target_grid` - Target voxel grid (map)
/// * `pose` - Transform to apply to source points
/// * `gauss` - Gaussian parameters for NDT score function
/// * `header` - ROS message header
/// * `config` - Visualization configuration
///
/// # Returns
/// PointCloud2 message with RGB-colored points
pub fn visualize_point_scores(
    source_points: &[[f32; 3]],
    target_grid: &VoxelGrid,
    pose: &nalgebra::Isometry3<f64>,
    gauss: &GaussianParams,
    header: &Header,
    config: &PointScoreConfig,
) -> PointCloud2 {
    // Compute per-point NVTL scores
    let nvtl_config = NvtlConfig {
        search_radius: config.search_radius,
        compute_per_point: true,
    };
    let nvtl_result = compute_nvtl(source_points, target_grid, pose, gauss, &nvtl_config);

    // Get per-point scores (or use empty vec if NVTL failed)
    let scores = nvtl_result.per_point_scores.unwrap_or_default();

    // Transform source points to world frame
    let transformed_points: Vec<[f32; 3]> = source_points
        .iter()
        .map(|p| {
            let pt = nalgebra::Point3::new(p[0] as f64, p[1] as f64, p[2] as f64);
            let transformed = pose * pt;
            [
                transformed.x as f32,
                transformed.y as f32,
                transformed.z as f32,
            ]
        })
        .collect();

    // Create PointCloud2 with XYZRGB fields
    let mut msg = PointCloud2::default();
    msg.header = header.clone();
    msg.height = 1;
    msg.width = source_points.len() as u32;
    msg.is_dense = true;
    msg.is_bigendian = false;

    // Define fields: x, y, z (float32), rgb (uint32)
    msg.fields = vec![
        PointField {
            name: "x".to_string(),
            offset: 0,
            datatype: PointField::FLOAT32,
            count: 1,
        },
        PointField {
            name: "y".to_string(),
            offset: 4,
            datatype: PointField::FLOAT32,
            count: 1,
        },
        PointField {
            name: "z".to_string(),
            offset: 8,
            datatype: PointField::FLOAT32,
            count: 1,
        },
        PointField {
            name: "rgb".to_string(),
            offset: 12,
            datatype: PointField::UINT32,
            count: 1,
        },
    ];

    msg.point_step = 16; // 3 floats (12 bytes) + 1 uint32 (4 bytes)
    msg.row_step = msg.point_step * msg.width;

    // Allocate data buffer
    let data_size = (msg.point_step * msg.width) as usize;
    msg.data = vec![0u8; data_size];

    // Fill point data
    for (i, (point, score)) in transformed_points.iter().zip(scores.iter()).enumerate() {
        let offset = (i * msg.point_step as usize) as usize;

        // Write XYZ
        msg.data[offset..offset + 4].copy_from_slice(&point[0].to_le_bytes());
        msg.data[offset + 4..offset + 8].copy_from_slice(&point[1].to_le_bytes());
        msg.data[offset + 8..offset + 12].copy_from_slice(&point[2].to_le_bytes());

        // Convert score to color
        let color = ndt_score_to_color(*score as f32, config.score_lower, config.score_upper);
        let rgb_packed = color_to_rgb_packed(&color);

        // Write RGB as uint32
        msg.data[offset + 12..offset + 16].copy_from_slice(&rgb_packed.to_le_bytes());
    }

    msg
}

// ============================================================================
// Particle Marker Visualization (Autoware Parity)
// ============================================================================

/// Color scheme for particle markers.
///
/// Autoware publishes markers in multiple namespaces with different color schemes:
/// - `initial_pose_transform_probability_color_marker` - by score
/// - `initial_pose_iteration_color_marker` - by iteration count
/// - `initial_pose_index_color_marker` - by particle index
#[derive(Clone, Copy, Debug)]
pub enum ParticleColorScheme {
    /// Green (high score) → Yellow → Red (low score)
    ByScore,
    /// Green (fast convergence) → Red (slow convergence)
    ByIteration,
    /// Rainbow gradient by particle index
    ByIndex,
}

/// Configuration for particle marker visualization.
#[derive(Clone, Debug)]
pub struct ParticleMarkerConfig {
    /// Marker lifetime in seconds (Autoware uses 10.0s)
    pub lifetime_sec: i32,
    /// Scale for initial pose markers
    pub initial_scale: f64,
    /// Scale for result pose markers
    pub result_scale: f64,
    /// Scale for best result marker
    pub best_scale: f64,
    /// Alpha for non-best markers
    pub alpha: f32,
    /// Alpha for best marker
    pub best_alpha: f32,
}

impl Default for ParticleMarkerConfig {
    fn default() -> Self {
        Self {
            lifetime_sec: 10,
            initial_scale: 0.15,
            result_scale: 0.2,
            best_scale: 0.4,
            alpha: 0.7,
            best_alpha: 1.0,
        }
    }
}

/// Convert HSV color to RGB.
///
/// # Arguments
/// * `h` - Hue in degrees [0, 360)
/// * `s` - Saturation [0, 1]
/// * `v` - Value [0, 1]
///
/// # Returns
/// (r, g, b) tuple with values in [0, 1]
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    if s <= 0.0 {
        return (v, v, v);
    }

    let h = h % 360.0;
    let h = h / 60.0;
    let i = h.floor() as i32;
    let f = h - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

/// Compute color for a particle based on the given color scheme.
fn particle_color(
    particle: &Particle,
    index: usize,
    total_particles: usize,
    min_score: f64,
    score_range: f64,
    max_iterations: i32,
    scheme: ParticleColorScheme,
    is_best: bool,
    config: &ParticleMarkerConfig,
) -> ColorRGBA {
    let alpha = if is_best {
        config.best_alpha
    } else {
        config.alpha
    };

    match scheme {
        ParticleColorScheme::ByScore => {
            // Green (high score) → Yellow → Red (low score)
            let normalized = ((particle.score - min_score) / score_range).clamp(0.0, 1.0);
            ColorRGBA {
                r: (1.0 - normalized) as f32,
                g: normalized as f32,
                b: 0.0,
                a: alpha,
            }
        }
        ParticleColorScheme::ByIteration => {
            // Green (fast: low iterations) → Red (slow: high iterations)
            let normalized = if max_iterations > 1 {
                (particle.iterations as f32) / (max_iterations as f32)
            } else {
                0.5
            };
            let normalized = normalized.clamp(0.0, 1.0);
            ColorRGBA {
                r: normalized,
                g: 1.0 - normalized,
                b: 0.0,
                a: alpha,
            }
        }
        ParticleColorScheme::ByIndex => {
            // Rainbow gradient: red → orange → yellow → green → cyan → blue → purple
            let hue = if total_particles > 1 {
                (index as f32) / (total_particles as f32) * 300.0 // Stop at purple (300°)
            } else {
                0.0
            };
            let (r, g, b) = hsv_to_rgb(hue, 1.0, 1.0);
            ColorRGBA { r, g, b, a: alpha }
        }
    }
}

/// Create a single sphere marker for a particle.
fn create_particle_sphere_marker(
    header: &Header,
    particle: &Particle,
    id: i32,
    namespace: &str,
    use_result_pose: bool,
    color: ColorRGBA,
    scale: f64,
    lifetime_sec: i32,
) -> Marker {
    let pose = if use_result_pose {
        particle.result_pose.clone()
    } else {
        particle.initial_pose.clone()
    };

    Marker {
        header: header.clone(),
        ns: namespace.to_string(),
        id,
        type_: Marker::SPHERE,
        action: Marker::ADD,
        pose,
        scale: Vector3 {
            x: scale,
            y: scale,
            z: scale,
        },
        color,
        lifetime: RosDuration {
            sec: lifetime_sec,
            nanosec: 0,
        },
        frame_locked: false,
        points: vec![],
        colors: vec![],
        texture_resource: String::new(),
        texture: sensor_msgs::msg::CompressedImage::default(),
        uv_coordinates: vec![],
        text: String::new(),
        mesh_resource: String::new(),
        mesh_file: visualization_msgs::msg::MeshFile::default(),
        mesh_use_embedded_materials: false,
    }
}

/// Create visualization markers for Monte Carlo particles with multiple color schemes.
///
/// This creates markers matching Autoware's particle visualization with multiple namespaces:
/// - `initial_pose_transform_probability_color_marker` - result poses colored by score
/// - `initial_pose_iteration_color_marker` - result poses colored by iteration count
/// - `initial_pose_index_color_marker` - result poses colored by particle index
/// - `result_pose_transform_probability_color_marker` - same as initial_pose but for results
/// - `result_pose_iteration_color_marker` - result pose iteration coloring
/// - `result_pose_index_color_marker` - result pose index coloring
/// - `monte_carlo_initial` - initial poses (blue spheres, for context)
///
/// # Arguments
/// * `header` - ROS message header
/// * `particles` - All evaluated particles
/// * `best_score` - Score of the best particle
/// * `config` - Marker configuration
///
/// # Returns
/// MarkerArray containing all particle markers
pub fn create_monte_carlo_markers_enhanced(
    header: &Header,
    particles: &[Particle],
    best_score: f64,
    config: &ParticleMarkerConfig,
) -> MarkerArray {
    if particles.is_empty() {
        return MarkerArray { markers: vec![] };
    }

    let mut markers = Vec::new();
    let mut id = 0;

    // Compute statistics for color normalization
    let min_score = particles
        .iter()
        .map(|p| p.score)
        .fold(f64::INFINITY, f64::min);
    let max_score = particles
        .iter()
        .map(|p| p.score)
        .fold(f64::NEG_INFINITY, f64::max);
    let score_range = (max_score - min_score).max(0.001);

    let max_iterations = particles.iter().map(|p| p.iterations).max().unwrap_or(30);
    let total_particles = particles.len();

    // Define all namespace/scheme combinations for result poses
    let result_schemes = [
        (
            "initial_pose_transform_probability_color_marker",
            ParticleColorScheme::ByScore,
        ),
        (
            "initial_pose_iteration_color_marker",
            ParticleColorScheme::ByIteration,
        ),
        (
            "initial_pose_index_color_marker",
            ParticleColorScheme::ByIndex,
        ),
        (
            "result_pose_transform_probability_color_marker",
            ParticleColorScheme::ByScore,
        ),
        (
            "result_pose_iteration_color_marker",
            ParticleColorScheme::ByIteration,
        ),
        (
            "result_pose_index_color_marker",
            ParticleColorScheme::ByIndex,
        ),
    ];

    // Create markers for each particle in each color scheme
    for (namespace, scheme) in &result_schemes {
        for (i, particle) in particles.iter().enumerate() {
            let is_best = (particle.score - best_score).abs() < 1e-10;
            let scale = if is_best {
                config.best_scale
            } else {
                config.result_scale
            };

            let color = particle_color(
                particle,
                i,
                total_particles,
                min_score,
                score_range,
                max_iterations,
                *scheme,
                is_best,
                config,
            );

            markers.push(create_particle_sphere_marker(
                header,
                particle,
                id,
                namespace,
                true, // use result_pose
                color,
                scale,
                config.lifetime_sec,
            ));
            id += 1;
        }
    }

    // Also create initial pose markers (blue spheres for context)
    for particle in particles {
        markers.push(create_particle_sphere_marker(
            header,
            particle,
            id,
            "monte_carlo_initial",
            false, // use initial_pose
            ColorRGBA {
                r: 0.3,
                g: 0.5,
                b: 1.0,
                a: 0.6,
            },
            config.initial_scale,
            config.lifetime_sec,
        ));
        id += 1;
    }

    MarkerArray { markers }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visualize_point_scores_empty() {
        let points: Vec<[f32; 3]> = vec![];
        let grid_config = ndt_cuda::VoxelGridConfig {
            resolution: 2.0,
            ..Default::default()
        };
        let grid = VoxelGrid::new(grid_config);
        let pose = nalgebra::Isometry3::identity();
        let gauss = GaussianParams::new(2.0, 0.55);
        let header = Header::default();
        let config = PointScoreConfig::default();

        let cloud = visualize_point_scores(&points, &grid, &pose, &gauss, &header, &config);
        assert_eq!(cloud.width, 0);
        assert!(cloud.data.is_empty());
    }

    #[test]
    fn test_visualize_point_scores_structure() {
        let points = vec![[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let grid_config = ndt_cuda::VoxelGridConfig {
            resolution: 2.0,
            ..Default::default()
        };
        let grid = VoxelGrid::new(grid_config);
        let pose = nalgebra::Isometry3::identity();
        let gauss = GaussianParams::new(2.0, 0.55);
        let header = Header::default();
        let config = PointScoreConfig::default();

        let cloud = visualize_point_scores(&points, &grid, &pose, &gauss, &header, &config);
        assert_eq!(cloud.width, 2);
        assert_eq!(cloud.height, 1);
        assert_eq!(cloud.point_step, 16);
        assert_eq!(cloud.fields.len(), 4);
        assert_eq!(cloud.fields[0].name, "x");
        assert_eq!(cloud.fields[3].name, "rgb");
    }

    // Helper to create test pose
    fn make_test_pose(x: f64, y: f64, z: f64) -> geometry_msgs::msg::Pose {
        geometry_msgs::msg::Pose {
            position: geometry_msgs::msg::Point { x, y, z },
            orientation: geometry_msgs::msg::Quaternion {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
        }
    }

    #[test]
    fn test_hsv_to_rgb() {
        // Red (0°)
        let (r, g, b) = hsv_to_rgb(0.0, 1.0, 1.0);
        assert!((r - 1.0).abs() < 0.01);
        assert!(g < 0.01);
        assert!(b < 0.01);

        // Green (120°)
        let (r, g, b) = hsv_to_rgb(120.0, 1.0, 1.0);
        assert!(r < 0.01);
        assert!((g - 1.0).abs() < 0.01);
        assert!(b < 0.01);

        // Blue (240°)
        let (r, g, b) = hsv_to_rgb(240.0, 1.0, 1.0);
        assert!(r < 0.01);
        assert!(g < 0.01);
        assert!((b - 1.0).abs() < 0.01);

        // Gray (no saturation)
        let (r, g, b) = hsv_to_rgb(0.0, 0.0, 0.5);
        assert!((r - 0.5).abs() < 0.01);
        assert!((g - 0.5).abs() < 0.01);
        assert!((b - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_create_monte_carlo_markers_enhanced_empty() {
        let particles: Vec<Particle> = vec![];
        let header = Header::default();
        let config = ParticleMarkerConfig::default();

        let markers = create_monte_carlo_markers_enhanced(&header, &particles, 0.0, &config);
        assert!(markers.markers.is_empty());
    }

    #[test]
    fn test_create_monte_carlo_markers_enhanced_namespaces() {
        let particles = vec![
            Particle::new(
                make_test_pose(0.0, 0.0, 0.0),
                make_test_pose(0.1, 0.0, 0.0),
                0.5,
                10,
            ),
            Particle::new(
                make_test_pose(1.0, 0.0, 0.0),
                make_test_pose(1.1, 0.0, 0.0),
                0.9,
                5,
            ),
            Particle::new(
                make_test_pose(2.0, 0.0, 0.0),
                make_test_pose(2.1, 0.0, 0.0),
                0.3,
                15,
            ),
        ];
        let header = Header::default();
        let config = ParticleMarkerConfig::default();
        let best_score = 0.9;

        let marker_array =
            create_monte_carlo_markers_enhanced(&header, &particles, best_score, &config);

        // 6 namespaces for result poses * 3 particles + 1 namespace for initial poses * 3 particles
        // = 18 + 3 = 21 markers
        assert_eq!(marker_array.markers.len(), 21);

        // Check that all expected namespaces are present
        let namespaces: std::collections::HashSet<_> =
            marker_array.markers.iter().map(|m| m.ns.as_str()).collect();

        assert!(namespaces.contains("initial_pose_transform_probability_color_marker"));
        assert!(namespaces.contains("initial_pose_iteration_color_marker"));
        assert!(namespaces.contains("initial_pose_index_color_marker"));
        assert!(namespaces.contains("result_pose_transform_probability_color_marker"));
        assert!(namespaces.contains("result_pose_iteration_color_marker"));
        assert!(namespaces.contains("result_pose_index_color_marker"));
        assert!(namespaces.contains("monte_carlo_initial"));
    }

    #[test]
    fn test_create_monte_carlo_markers_enhanced_lifetime() {
        let particles = vec![Particle::new(
            make_test_pose(0.0, 0.0, 0.0),
            make_test_pose(0.1, 0.0, 0.0),
            0.5,
            10,
        )];
        let header = Header::default();
        let config = ParticleMarkerConfig {
            lifetime_sec: 10,
            ..Default::default()
        };

        let marker_array = create_monte_carlo_markers_enhanced(&header, &particles, 0.5, &config);

        // All markers should have 10 second lifetime
        for marker in &marker_array.markers {
            assert_eq!(marker.lifetime.sec, 10);
        }
    }

    #[test]
    fn test_particle_color_by_score() {
        let particle = Particle::new(
            make_test_pose(0.0, 0.0, 0.0),
            make_test_pose(0.0, 0.0, 0.0),
            1.0, // max score
            10,
        );
        let config = ParticleMarkerConfig::default();

        // High score should be green
        let color = particle_color(
            &particle,
            0,
            1,
            0.0, // min_score
            1.0, // score_range
            10,  // max_iterations
            ParticleColorScheme::ByScore,
            false,
            &config,
        );
        assert!(color.g > color.r); // Green > Red for high score
    }

    #[test]
    fn test_particle_color_by_iteration() {
        let fast_particle = Particle::new(
            make_test_pose(0.0, 0.0, 0.0),
            make_test_pose(0.0, 0.0, 0.0),
            0.5,
            1, // very fast
        );
        let slow_particle = Particle::new(
            make_test_pose(0.0, 0.0, 0.0),
            make_test_pose(0.0, 0.0, 0.0),
            0.5,
            30, // slow
        );
        let config = ParticleMarkerConfig::default();

        let fast_color = particle_color(
            &fast_particle,
            0,
            1,
            0.0,
            1.0,
            30,
            ParticleColorScheme::ByIteration,
            false,
            &config,
        );
        let slow_color = particle_color(
            &slow_particle,
            0,
            1,
            0.0,
            1.0,
            30,
            ParticleColorScheme::ByIteration,
            false,
            &config,
        );

        // Fast (low iterations) should be more green
        assert!(fast_color.g > slow_color.g);
        // Slow (high iterations) should be more red
        assert!(slow_color.r > fast_color.r);
    }
}
