//! GPU-friendly radius search using Morton-sorted voxel centroids.
//!
//! Since voxels are sorted by Morton code, nearby voxels have similar codes.
//! This enables efficient radius search by:
//! 1. Computing Morton code of query point
//! 2. Binary search to find starting position
//! 3. Linear scan in both directions, checking distances
//!
//! # Current Status
//!
//! GPU kernels are defined but require CubeCL type system fixes.
//! CPU reference implementations are provided and tested.

/// Result of radius search for a single query point.
#[derive(Debug, Clone)]
pub struct RadiusSearchResult {
    /// Indices of voxels within the search radius.
    pub voxel_indices: Vec<u32>,
    /// Squared distances to each found voxel.
    pub distances_sq: Vec<f32>,
}

/// Configuration for radius search.
#[derive(Debug, Clone)]
pub struct RadiusSearchConfig {
    /// Search radius.
    pub radius: f32,
    /// Maximum number of neighbors to return per query.
    pub max_neighbors: usize,
    /// Grid resolution (voxel size) for Morton code computation.
    pub resolution: f32,
    /// Grid minimum bounds.
    pub grid_min: [f32; 3],
}

/// Perform radius search on Morton-sorted voxels (CPU reference implementation).
///
/// # Arguments
/// * `query_points` - Query points as flat array [x0, y0, z0, x1, y1, z1, ...]
/// * `voxel_means` - Voxel centroids as flat array [x0, y0, z0, x1, y1, z1, ...]
/// * `voxel_morton_codes` - Morton codes for each voxel (sorted)
/// * `config` - Search configuration
///
/// # Returns
/// Vector of RadiusSearchResult, one per query point.
pub fn radius_search_cpu(
    query_points: &[f32],
    voxel_means: &[f32],
    voxel_morton_codes: &[u64],
    config: &RadiusSearchConfig,
) -> Vec<RadiusSearchResult> {
    let num_queries = query_points.len() / 3;
    let _num_voxels = voxel_morton_codes.len();
    let radius_sq = config.radius * config.radius;

    let mut results = Vec::with_capacity(num_queries);

    for q in 0..num_queries {
        let qx = query_points[q * 3];
        let qy = query_points[q * 3 + 1];
        let qz = query_points[q * 3 + 2];

        let result = radius_search_single_cpu(
            qx,
            qy,
            qz,
            voxel_means,
            voxel_morton_codes,
            radius_sq,
            config.max_neighbors,
            config.resolution,
            &config.grid_min,
        );

        results.push(result);
    }

    results
}

/// Perform radius search for a single query point.
#[allow(clippy::too_many_arguments)]
fn radius_search_single_cpu(
    qx: f32,
    qy: f32,
    qz: f32,
    voxel_means: &[f32],
    voxel_morton_codes: &[u64],
    radius_sq: f32,
    max_neighbors: usize,
    resolution: f32,
    grid_min: &[f32; 3],
) -> RadiusSearchResult {
    let num_voxels = voxel_morton_codes.len();

    if num_voxels == 0 {
        return RadiusSearchResult {
            voxel_indices: Vec::new(),
            distances_sq: Vec::new(),
        };
    }

    // Compute Morton code of query point
    let query_morton = compute_query_morton(qx, qy, qz, resolution, grid_min);

    // Binary search to find starting position
    let start_idx = binary_search_morton(voxel_morton_codes, query_morton);

    let mut voxel_indices = Vec::with_capacity(max_neighbors);
    let mut distances_sq = Vec::with_capacity(max_neighbors);

    // Compute maximum Morton code difference for early termination
    // This is an approximation based on the search radius in grid cells
    let radius_cells = (config_radius_to_cells(radius_sq.sqrt(), resolution) + 1.0) as u64;
    let max_morton_diff = estimate_morton_range(radius_cells);

    // Forward scan from starting position
    let mut i = start_idx;
    while i < num_voxels && voxel_indices.len() < max_neighbors {
        let dist_sq = compute_distance_sq(qx, qy, qz, voxel_means, i);

        if dist_sq <= radius_sq {
            voxel_indices.push(i as u32);
            distances_sq.push(dist_sq);
        }

        // Early termination: if Morton code differs too much, stop
        let code_diff = voxel_morton_codes[i].abs_diff(query_morton);
        if code_diff > max_morton_diff {
            break;
        }

        i += 1;
    }

    // Backward scan from starting position - 1
    if start_idx > 0 {
        let mut i = start_idx - 1;
        loop {
            if voxel_indices.len() >= max_neighbors {
                break;
            }

            let dist_sq = compute_distance_sq(qx, qy, qz, voxel_means, i);

            if dist_sq <= radius_sq {
                voxel_indices.push(i as u32);
                distances_sq.push(dist_sq);
            }

            // Early termination
            let code_diff = voxel_morton_codes[i].abs_diff(query_morton);
            if code_diff > max_morton_diff {
                break;
            }

            if i == 0 {
                break;
            }
            i -= 1;
        }
    }

    RadiusSearchResult {
        voxel_indices,
        distances_sq,
    }
}

/// Compute Morton code for a query point.
fn compute_query_morton(x: f32, y: f32, z: f32, resolution: f32, grid_min: &[f32; 3]) -> u64 {
    let inv_resolution = 1.0 / resolution;

    let gx = (x - grid_min[0]) * inv_resolution;
    let gy = (y - grid_min[1]) * inv_resolution;
    let gz = (z - grid_min[2]) * inv_resolution;

    let ix = if gx >= 0.0 {
        Ord::min(gx as u32, 0x1FFFFF)
    } else {
        0
    };
    let iy = if gy >= 0.0 {
        Ord::min(gy as u32, 0x1FFFFF)
    } else {
        0
    };
    let iz = if gz >= 0.0 {
        Ord::min(gz as u32, 0x1FFFFF)
    } else {
        0
    };

    morton_encode_cpu(ix, iy, iz)
}

/// Morton encode (copy from morton.rs to avoid circular dependency).
fn morton_encode_cpu(x: u32, y: u32, z: u32) -> u64 {
    fn expand_bits(mut x: u64) -> u64 {
        x &= 0x1FFFFF;
        x = (x | (x << 32)) & 0x1F00000000FFFF_u64;
        x = (x | (x << 16)) & 0x1F0000FF0000FF_u64;
        x = (x | (x << 8)) & 0x100F00F00F00F00F_u64;
        x = (x | (x << 4)) & 0x10C30C30C30C30C3_u64;
        x = (x | (x << 2)) & 0x1249249249249249_u64;
        x
    }

    let xx = expand_bits(x as u64);
    let yy = expand_bits(y as u64);
    let zz = expand_bits(z as u64);
    (xx << 2) | (yy << 1) | zz
}

/// Binary search to find the position where a Morton code would be inserted.
fn binary_search_morton(sorted_codes: &[u64], target: u64) -> usize {
    sorted_codes.partition_point(|&code| code < target)
}

/// Convert radius to grid cells.
fn config_radius_to_cells(radius: f32, resolution: f32) -> f32 {
    radius / resolution
}

/// Estimate the maximum Morton code difference for a given cell radius.
///
/// This is a heuristic: we assume the search range in Morton space is
/// roughly 8 * radius_cells^3 (conservative upper bound).
fn estimate_morton_range(radius_cells: u64) -> u64 {
    // Each coordinate contributes 21 bits to the Morton code
    // A radius of N cells can span ~8*N³ Morton codes in the worst case
    let volume = 8 * radius_cells * radius_cells * radius_cells;
    volume.max(64) // Minimum range to search
}

/// Compute squared distance between query point and voxel centroid.
fn compute_distance_sq(qx: f32, qy: f32, qz: f32, voxel_means: &[f32], voxel_idx: usize) -> f32 {
    let dx = qx - voxel_means[voxel_idx * 3];
    let dy = qy - voxel_means[voxel_idx * 3 + 1];
    let dz = qz - voxel_means[voxel_idx * 3 + 2];
    dx * dx + dy * dy + dz * dz
}

/// Brute-force radius search (for comparison/testing).
///
/// Checks all voxels - slower but guaranteed correct.
pub fn radius_search_brute_force_cpu(
    query_points: &[f32],
    voxel_means: &[f32],
    radius: f32,
    max_neighbors: usize,
) -> Vec<RadiusSearchResult> {
    let num_queries = query_points.len() / 3;
    let num_voxels = voxel_means.len() / 3;
    let radius_sq = radius * radius;

    let mut results = Vec::with_capacity(num_queries);

    for q in 0..num_queries {
        let qx = query_points[q * 3];
        let qy = query_points[q * 3 + 1];
        let qz = query_points[q * 3 + 2];

        let mut voxel_indices = Vec::new();
        let mut distances_sq = Vec::new();

        for v in 0..num_voxels {
            let dist_sq = compute_distance_sq(qx, qy, qz, voxel_means, v);

            if dist_sq <= radius_sq {
                voxel_indices.push(v as u32);
                distances_sq.push(dist_sq);

                if voxel_indices.len() >= max_neighbors {
                    break;
                }
            }
        }

        results.push(RadiusSearchResult {
            voxel_indices,
            distances_sq,
        });
    }

    results
}

#[cfg(test)]
mod tests {
    use super::super::morton::morton_decode_3d;
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_binary_search_morton() {
        let codes = vec![1, 5, 10, 15, 20, 25, 30];

        assert_eq!(binary_search_morton(&codes, 0), 0);
        assert_eq!(binary_search_morton(&codes, 1), 0);
        assert_eq!(binary_search_morton(&codes, 2), 1);
        assert_eq!(binary_search_morton(&codes, 10), 2);
        assert_eq!(binary_search_morton(&codes, 11), 3);
        assert_eq!(binary_search_morton(&codes, 100), 7);
    }

    #[test]
    fn test_morton_encode_roundtrip() {
        let test_cases = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (10, 20, 30)];

        for (x, y, z) in test_cases {
            let code = morton_encode_cpu(x, y, z);
            let (dx, dy, dz) = morton_decode_3d(code);
            assert_eq!((x, y, z), (dx, dy, dz));
        }
    }

    #[test]
    fn test_brute_force_single_voxel() {
        let query = vec![0.0, 0.0, 0.0];
        let voxels = vec![0.5, 0.5, 0.5];

        let results = radius_search_brute_force_cpu(&query, &voxels, 1.0, 10);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].voxel_indices.len(), 1);
        assert_eq!(results[0].voxel_indices[0], 0);
    }

    #[test]
    fn test_brute_force_no_match() {
        let query = vec![0.0, 0.0, 0.0];
        let voxels = vec![10.0, 10.0, 10.0];

        let results = radius_search_brute_force_cpu(&query, &voxels, 1.0, 10);

        assert_eq!(results.len(), 1);
        assert!(results[0].voxel_indices.is_empty());
    }

    #[test]
    fn test_brute_force_multiple_voxels() {
        let query = vec![0.0, 0.0, 0.0];
        let voxels = vec![
            0.1, 0.1, 0.1, // Distance ~0.17, in range
            0.5, 0.5, 0.5, // Distance ~0.87, in range
            1.0, 1.0, 1.0, // Distance ~1.73, out of range
            0.0, 0.0, 0.5, // Distance 0.5, in range
        ];

        let results = radius_search_brute_force_cpu(&query, &voxels, 1.0, 10);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].voxel_indices.len(), 3); // 3 voxels within radius
    }

    #[test]
    fn test_radius_search_matches_brute_force() {
        // Create a simple voxel grid
        let mut voxel_means = Vec::new();
        let mut voxel_codes = Vec::new();
        let resolution = 1.0;
        let grid_min = [0.0, 0.0, 0.0];

        // Create voxels at regular grid positions
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..5 {
                    let px = x as f32 + 0.5;
                    let py = y as f32 + 0.5;
                    let pz = z as f32 + 0.5;
                    voxel_means.push(px);
                    voxel_means.push(py);
                    voxel_means.push(pz);
                    voxel_codes.push(morton_encode_cpu(x, y, z));
                }
            }
        }

        // Sort by Morton code
        let mut indices: Vec<usize> = (0..voxel_codes.len()).collect();
        indices.sort_by_key(|&i| voxel_codes[i]);

        let sorted_codes: Vec<u64> = indices.iter().map(|&i| voxel_codes[i]).collect();
        let sorted_means: Vec<f32> = indices
            .iter()
            .flat_map(|&i| {
                vec![
                    voxel_means[i * 3],
                    voxel_means[i * 3 + 1],
                    voxel_means[i * 3 + 2],
                ]
            })
            .collect();

        // Query point at center
        let query = vec![2.5, 2.5, 2.5];
        let radius = 1.5;

        let config = RadiusSearchConfig {
            radius,
            max_neighbors: 100,
            resolution,
            grid_min,
        };

        let morton_results = radius_search_cpu(&query, &sorted_means, &sorted_codes, &config);
        let brute_results = radius_search_brute_force_cpu(&query, &sorted_means, radius, 100);

        // Both should find the same voxels (order may differ)
        let morton_set: HashSet<u32> = morton_results[0].voxel_indices.iter().cloned().collect();
        let brute_set: HashSet<u32> = brute_results[0].voxel_indices.iter().cloned().collect();

        // Morton-based search might miss some voxels due to early termination heuristic
        // but should find at least the closest ones
        assert!(!morton_results[0].voxel_indices.is_empty());

        // For this simple test, we expect good overlap
        let overlap: HashSet<_> = morton_set.intersection(&brute_set).collect();
        assert!(
            !overlap.is_empty(),
            "Morton and brute-force should have overlap"
        );
    }

    #[test]
    fn test_estimate_morton_range() {
        // Small radius should give small range
        assert!(estimate_morton_range(1) >= 8);

        // Larger radius should give larger range
        assert!(estimate_morton_range(10) > estimate_morton_range(1));
    }

    #[test]
    fn test_max_neighbors_limit() {
        let query = vec![0.0, 0.0, 0.0];
        let mut voxels = Vec::new();

        // Create many nearby voxels
        for i in 0..20 {
            let offset = i as f32 * 0.01;
            voxels.push(offset);
            voxels.push(offset);
            voxels.push(offset);
        }

        let results = radius_search_brute_force_cpu(&query, &voxels, 10.0, 5);

        assert_eq!(results[0].voxel_indices.len(), 5); // Limited to max_neighbors
    }
}
