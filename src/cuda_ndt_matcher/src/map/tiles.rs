//! Map tile storage and caching.
//!
//! This module manages the point cloud map used for NDT matching:
//! - Stores map tiles by ID
//! - Triggers updates when vehicle moves beyond threshold
//! - Filters map points within radius of current position
//! - Combines tiles into target cloud for NDT
//!
//! ## Autoware Compatibility
//!
//! This implementation uses `GetDifferentialPointCloudMap` service for differential
//! tile loading, matching Autoware's `MapUpdateModule` behavior:
//!
//! | Feature | Autoware | This Implementation |
//! |---------|----------|---------------------|
//! | Map Loading | GetDifferentialPointCloudMap service | Same ✓ |
//! | Tile Management | Per-tile add/remove via service | Same ✓ |
//! | Update Trigger | Timer callback | Position-based check on each alignment |
//! | Secondary NDT | Yes (non-blocking updates) | No (direct update with lock) |

use crate::io::params::DynamicMapParams;
use geometry_msgs::msg::Point;
use parking_lot::RwLock;
use rclrs::log_debug;
use std::collections::HashMap;
use std::time::Instant;

pub(super) const LOGGER_NAME: &str = "ndt_scan_matcher.map_module";

/// A map tile with its point cloud data
#[derive(Debug, Clone)]
pub(crate) struct MapTile {
    /// Unique identifier for this tile
    pub(crate) id: String,
    /// Center position of the tile
    #[allow(dead_code)] // Tile metadata; used indirectly via Arc patterns
    pub(crate) center: Point,
    /// Point cloud data as [x, y, z] points
    pub(crate) points: Vec<[f32; 3]>,
}

/// Result of a map update check
#[derive(Debug, Clone)]
pub(crate) struct MapUpdateResult {
    /// Whether the map was updated
    pub(crate) updated: bool,
    /// Number of tiles currently loaded
    pub(crate) tiles_loaded: usize,
    /// Total points in combined map
    pub(crate) total_points: usize,
    /// Distance from last update position
    pub(crate) distance_from_last_update: f64,
    /// Time taken for map update (if updated)
    #[allow(dead_code)] // Monitoring field; logged but not read in code
    pub(crate) update_time_ms: f64,
}

/// Manages dynamic map loading and caching
pub(crate) struct MapUpdateModule {
    /// Map tiles indexed by ID
    tiles: RwLock<HashMap<String, MapTile>>,
    /// Last position where map was updated
    last_update_position: RwLock<Option<Point>>,
    /// Combined map points (filtered by radius)
    cached_map_points: RwLock<Vec<[f32; 3]>>,
    /// Parameters for map loading
    params: DynamicMapParams,
    /// Flag indicating map needs rebuild
    needs_rebuild: RwLock<bool>,
}

impl MapUpdateModule {
    /// Create a new map update module
    pub(crate) fn new(params: DynamicMapParams) -> Self {
        Self {
            tiles: RwLock::new(HashMap::new()),
            last_update_position: RwLock::new(None),
            cached_map_points: RwLock::new(Vec::new()),
            params,
            needs_rebuild: RwLock::new(true),
        }
    }

    /// Check if map update is needed based on current position
    ///
    /// Returns true if:
    /// - No tiles are loaded (initial map load)
    /// - No previous update position exists (first update)
    /// - Distance from last update exceeds update_distance threshold
    pub(crate) fn should_update(&self, current_position: &Point) -> bool {
        // Always update if no tiles are loaded (need initial map)
        if self.tiles.read().is_empty() {
            return true;
        }

        let last_pos = self.last_update_position.read();

        match last_pos.as_ref() {
            None => true, // First update
            Some(last) => {
                let distance = euclidean_distance_2d(current_position, last);
                distance > self.params.update_distance
            }
        }
    }

    /// Get distance from last update position
    pub(crate) fn distance_from_last_update(&self, current_position: &Point) -> f64 {
        let last_pos = self.last_update_position.read();
        match last_pos.as_ref() {
            None => f64::INFINITY,
            Some(last) => euclidean_distance_2d(current_position, last),
        }
    }

    /// Check if position is out of current map range
    ///
    /// Returns true if the current position plus lidar radius
    /// exceeds the map radius from the last update position
    pub(crate) fn out_of_map_range(&self, current_position: &Point) -> bool {
        let last_pos = self.last_update_position.read();

        match last_pos.as_ref() {
            None => true,
            Some(last) => {
                let distance = euclidean_distance_2d(current_position, last);
                distance + self.params.lidar_radius > self.params.map_radius
            }
        }
    }

    /// Add or update a map tile
    pub(crate) fn add_tile(&self, tile: MapTile) {
        let mut tiles = self.tiles.write();
        tiles.insert(tile.id.clone(), tile);
        *self.needs_rebuild.write() = true;
    }

    /// Remove a map tile by ID
    pub(crate) fn remove_tile(&self, tile_id: &str) -> bool {
        let mut tiles = self.tiles.write();
        let removed = tiles.remove(tile_id).is_some();
        if removed {
            *self.needs_rebuild.write() = true;
        }
        removed
    }

    /// Get list of currently loaded tile IDs
    pub(crate) fn get_loaded_tile_ids(&self) -> Vec<String> {
        self.tiles.read().keys().cloned().collect()
    }

    /// Get number of loaded tiles
    pub(crate) fn tile_count(&self) -> usize {
        self.tiles.read().len()
    }

    /// Update the map based on current position
    ///
    /// This filters points within map_radius of the current position
    /// and rebuilds the cached map points.
    ///
    /// This method implements Autoware's `should_update_map` + `update_map` logic:
    /// - Checks if position has moved beyond `update_distance`
    /// - Checks if we're approaching the edge of the loaded map
    /// - Filters points within `map_radius` of current position
    ///
    /// Returns update result with statistics
    pub(crate) fn update_map(&self, current_position: &Point) -> MapUpdateResult {
        let distance = self.distance_from_last_update(current_position);
        let needs_rebuild = *self.needs_rebuild.read();

        // Check if we need to update (matches Autoware's should_update_map)
        if !needs_rebuild && !self.should_update(current_position) {
            let cached = self.cached_map_points.read();
            return MapUpdateResult {
                updated: false,
                tiles_loaded: self.tile_count(),
                total_points: cached.len(),
                distance_from_last_update: distance,
                update_time_ms: 0.0,
            };
        }

        let start_time = Instant::now();

        // Check if we're falling behind (Autoware's "dynamic map loading not keeping up" check)
        if distance + self.params.lidar_radius > self.params.map_radius {
            log_debug!(
                LOGGER_NAME,
                "Map update not keeping up: distance={:.1}m + lidar_radius={:.1}m > map_radius={:.1}m",
                distance,
                self.params.lidar_radius,
                self.params.map_radius
            );
        }

        // Combine all points from loaded tiles (matching Autoware's behavior).
        // Note: Unlike CUDA's previous implementation, we do NOT filter by distance from
        // current position. Autoware uses all points from loaded tiles, and the tile
        // loading service already handles the spatial selection via map_radius.
        let tiles = self.tiles.read();
        let total_points: usize = tiles.values().map(|t| t.points.len()).sum();
        let mut combined_points = Vec::with_capacity(total_points);

        for tile in tiles.values() {
            combined_points.extend_from_slice(&tile.points);
        }

        let tiles_loaded = tiles.len();
        drop(tiles);

        // Update cached map
        *self.cached_map_points.write() = combined_points;
        *self.last_update_position.write() = Some(current_position.clone());
        *self.needs_rebuild.write() = false;

        let update_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        log_debug!(
            LOGGER_NAME,
            "Map updated: {} points from {} tiles (took {:.1}ms)",
            total_points,
            tiles_loaded,
            update_time_ms
        );

        MapUpdateResult {
            updated: true,
            tiles_loaded,
            total_points,
            distance_from_last_update: distance,
            update_time_ms,
        }
    }

    /// Check and update map if needed, returning whether NDT target needs updating
    ///
    /// This is a convenience method that combines `should_update` and `update_map`,
    /// returning the filtered map points if an update was performed.
    ///
    /// # Returns
    /// - `Some(points)` if map was updated and NDT target should be refreshed
    /// - `None` if no update was needed
    pub(crate) fn check_and_update(&self, current_position: &Point) -> Option<Vec<[f32; 3]>> {
        let result = self.update_map(current_position);
        if result.updated {
            self.get_map_points()
        } else {
            None
        }
    }

    /// Get the current cached map points
    ///
    /// Returns None if no map is loaded
    pub(crate) fn get_map_points(&self) -> Option<Vec<[f32; 3]>> {
        let cached = self.cached_map_points.read();
        if cached.is_empty() {
            None
        } else {
            Some(cached.clone())
        }
    }

    /// Clear all map data
    pub(crate) fn clear(&self) {
        self.tiles.write().clear();
        self.cached_map_points.write().clear();
        *self.last_update_position.write() = None;
        *self.needs_rebuild.write() = true;
    }

    /// Load a complete map (replaces all tiles)
    ///
    /// This is a convenience method for loading a single large map
    /// without tile management.
    pub(crate) fn load_full_map(&self, points: Vec<[f32; 3]>) {
        let center = if points.is_empty() {
            Point {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            }
        } else {
            // Calculate centroid
            let mut sum_x = 0.0f64;
            let mut sum_y = 0.0f64;
            let mut sum_z = 0.0f64;
            for p in &points {
                sum_x += p[0] as f64;
                sum_y += p[1] as f64;
                sum_z += p[2] as f64;
            }
            let n = points.len() as f64;
            Point {
                x: sum_x / n,
                y: sum_y / n,
                z: sum_z / n,
            }
        };

        let tile = MapTile {
            id: "full_map".to_string(),
            center,
            points,
        };

        self.clear();
        self.add_tile(tile);
    }
}

/// Calculate 2D Euclidean distance between two points
fn euclidean_distance_2d(a: &Point, b: &Point) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_params() -> DynamicMapParams {
        DynamicMapParams {
            update_distance: 20.0,
            map_radius: 150.0,
            lidar_radius: 100.0,
        }
    }

    fn make_point(x: f64, y: f64, z: f64) -> Point {
        Point { x, y, z }
    }

    #[test]
    fn test_should_update_first_time() {
        let module = MapUpdateModule::new(make_params());
        let pos = make_point(0.0, 0.0, 0.0);

        // First update should always return true
        assert!(module.should_update(&pos));
    }

    #[test]
    fn test_should_update_within_threshold() {
        let module = MapUpdateModule::new(make_params());
        let pos1 = make_point(0.0, 0.0, 0.0);
        let pos2 = make_point(10.0, 0.0, 0.0); // 10m away, below 20m threshold

        // Add a tile so we're testing distance threshold, not empty-tiles case
        let tile = MapTile {
            id: "tile_1".to_string(),
            center: make_point(0.0, 0.0, 0.0),
            points: vec![[0.0, 0.0, 0.0]],
        };
        module.add_tile(tile);

        module.update_map(&pos1);
        assert!(!module.should_update(&pos2));
    }

    #[test]
    fn test_should_update_beyond_threshold() {
        let module = MapUpdateModule::new(make_params());
        let pos1 = make_point(0.0, 0.0, 0.0);
        let pos2 = make_point(25.0, 0.0, 0.0); // 25m away, above 20m threshold

        // Add a tile so we're testing distance threshold, not empty-tiles case
        let tile = MapTile {
            id: "tile_1".to_string(),
            center: make_point(0.0, 0.0, 0.0),
            points: vec![[0.0, 0.0, 0.0]],
        };
        module.add_tile(tile);

        module.update_map(&pos1);
        assert!(module.should_update(&pos2));
    }

    #[test]
    fn test_should_update_empty_tiles() {
        let module = MapUpdateModule::new(make_params());
        let pos1 = make_point(0.0, 0.0, 0.0);
        let pos2 = make_point(5.0, 0.0, 0.0); // Within threshold

        // With no tiles, should_update always returns true (need initial map load)
        module.update_map(&pos1);
        assert!(
            module.should_update(&pos2),
            "should update when no tiles loaded"
        );
    }

    #[test]
    fn test_out_of_map_range() {
        let module = MapUpdateModule::new(make_params());
        let pos1 = make_point(0.0, 0.0, 0.0);

        // Add a tile so we have a map
        let tile = MapTile {
            id: "tile_1".to_string(),
            center: make_point(0.0, 0.0, 0.0),
            points: vec![[0.0, 0.0, 0.0]],
        };
        module.add_tile(tile);

        module.update_map(&pos1);

        // Position at 40m: 40 + 100 (lidar) = 140 < 150 (map_radius)
        let pos_safe = make_point(40.0, 0.0, 0.0);
        assert!(!module.out_of_map_range(&pos_safe));

        // Position at 60m: 60 + 100 (lidar) = 160 > 150 (map_radius)
        let pos_out = make_point(60.0, 0.0, 0.0);
        assert!(module.out_of_map_range(&pos_out));
    }

    #[test]
    fn test_add_remove_tiles() {
        let module = MapUpdateModule::new(make_params());

        let tile = MapTile {
            id: "tile_1".to_string(),
            center: make_point(0.0, 0.0, 0.0),
            points: vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        };

        module.add_tile(tile);
        assert_eq!(module.tile_count(), 1);
        assert!(module.get_loaded_tile_ids().contains(&"tile_1".to_string()));

        module.remove_tile("tile_1");
        assert_eq!(module.tile_count(), 0);
    }

    #[test]
    fn test_map_radius_filtering() {
        // Note: MapUpdateModule does NOT filter points by radius.
        // It uses all points from loaded tiles because the tile loading service
        // (external to this module) handles spatial selection via map_radius.
        // This matches Autoware's behavior.
        let mut params = make_params();
        params.map_radius = 10.0;

        let module = MapUpdateModule::new(params);

        let tile = MapTile {
            id: "tile_1".to_string(),
            center: make_point(0.0, 0.0, 0.0),
            points: vec![
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [15.0, 0.0, 0.0],
                [0.0, 20.0, 0.0],
            ],
        };

        module.add_tile(tile);

        let result = module.update_map(&make_point(0.0, 0.0, 0.0));

        assert!(result.updated);
        // All 4 points are included - no radius filtering at this level
        assert_eq!(result.total_points, 4);
        assert!(result.update_time_ms >= 0.0);
    }

    #[test]
    fn test_check_and_update() {
        let mut params = make_params();
        params.update_distance = 5.0;

        let module = MapUpdateModule::new(params);

        let tile = MapTile {
            id: "tile_1".to_string(),
            center: make_point(0.0, 0.0, 0.0),
            points: vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        };
        module.add_tile(tile);

        // First call should update
        let points = module.check_and_update(&make_point(0.0, 0.0, 0.0));
        assert!(points.is_some());

        // Second call at same position should not update
        let points = module.check_and_update(&make_point(0.0, 0.0, 0.0));
        assert!(points.is_none());

        // Call at distant position should update
        let points = module.check_and_update(&make_point(10.0, 0.0, 0.0));
        assert!(points.is_some());
    }

    #[test]
    fn test_load_full_map() {
        let module = MapUpdateModule::new(make_params());

        let points = vec![[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0]];

        module.load_full_map(points.clone());

        assert_eq!(module.tile_count(), 1);
        assert!(
            module
                .get_loaded_tile_ids()
                .contains(&"full_map".to_string())
        );
    }
}
