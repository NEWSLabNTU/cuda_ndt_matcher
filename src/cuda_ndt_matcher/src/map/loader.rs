//! Dynamic map loader using GetDifferentialPointCloudMap service.

use super::tiles::{MapTile, MapUpdateModule, LOGGER_NAME};
use crate::io::pointcloud;
use autoware_map_msgs::msg::{AreaInfo, PointCloudMapCellWithID};
use autoware_map_msgs::srv::{GetDifferentialPointCloudMap, GetDifferentialPointCloudMap_Request};
use geometry_msgs::msg::Point;
use parking_lot::RwLock;
use rclrs::{log_debug, log_error, log_info, log_warn, Client, Node};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Status of the last map loader request.
#[derive(Debug, Clone, Default)]
pub(crate) struct MapLoaderStatus {
    /// Whether the pcd_loader service is available
    pub(crate) service_available: bool,
    /// Whether the last request was successful
    pub(crate) last_request_success: bool,
    /// Number of tiles added in last update
    pub(crate) tiles_added: usize,
    /// Number of tiles removed in last update
    pub(crate) tiles_removed: usize,
    /// Points added in last update
    pub(crate) points_added: usize,
    /// Time of last update (epoch seconds)
    pub(crate) last_update_time: f64,
    /// Error message if last request failed
    pub(crate) error_message: Option<String>,
}

/// Dynamic map loader using GetDifferentialPointCloudMap service.
///
/// This implements Autoware's differential map loading pattern:
/// 1. Requests map tiles around the current position via service
/// 2. Receives new tiles to add and old tile IDs to remove
/// 3. Updates the MapUpdateModule with the differential changes
///
/// The service client requires the node to spin for callbacks to work.
pub(crate) struct DynamicMapLoader {
    /// Service client for map loading
    client: Client<GetDifferentialPointCloudMap>,
    /// Map update module to populate
    map_module: Arc<MapUpdateModule>,
    /// Flag indicating a request is in flight
    request_pending: Arc<AtomicBool>,
    /// Status of the loader
    status: Arc<RwLock<MapLoaderStatus>>,
}

impl DynamicMapLoader {
    /// Create a new dynamic map loader.
    ///
    /// # Arguments
    /// * `node` - The ROS node to create the client on
    /// * `service_name` - Name of the pcd_loader_service (typically "pcd_loader_service")
    /// * `map_module` - The map module to populate with loaded tiles
    pub(crate) fn new(
        node: &Node,
        service_name: &str,
        map_module: Arc<MapUpdateModule>,
    ) -> Result<Self, rclrs::RclrsError> {
        let client = node.create_client::<GetDifferentialPointCloudMap>(service_name)?;
        log_info!(
            LOGGER_NAME,
            "Created GetDifferentialPointCloudMap client for '{service_name}'"
        );

        Ok(Self {
            client,
            map_module,
            request_pending: Arc::new(AtomicBool::new(false)),
            status: Arc::new(RwLock::new(MapLoaderStatus::default())),
        })
    }

    /// Check if the map service is available.
    pub(crate) fn service_is_ready(&self) -> bool {
        let is_ready = self.client.service_is_ready().unwrap_or(false);
        self.status.write().service_available = is_ready;
        is_ready
    }

    /// Check if a request is currently pending.
    #[allow(dead_code)] // Monitoring API; reserved for future diagnostics
    pub(crate) fn is_request_pending(&self) -> bool {
        self.request_pending.load(Ordering::SeqCst)
    }

    /// Get the current loader status.
    pub(crate) fn get_status(&self) -> MapLoaderStatus {
        self.status.read().clone()
    }

    /// Request differential map update around a position.
    ///
    /// This is an async operation - the callback will be invoked when the
    /// response arrives. The node must spin for the callback to be processed.
    ///
    /// # Arguments
    /// * `position` - Current position to request map around
    /// * `map_radius` - Radius of area to load (typically from DynamicMapParams)
    ///
    /// # Returns
    /// * `Ok(true)` - Request was sent successfully
    /// * `Ok(false)` - Request not sent (service unavailable or request pending)
    /// * `Err(_)` - Error sending request
    pub(crate) fn request_map_update(
        &self,
        position: &Point,
        map_radius: f32,
    ) -> Result<bool, rclrs::RclrsError> {
        // Check if service is available
        if !self.service_is_ready() {
            let mut status = self.status.write();
            status.service_available = false;
            status.error_message = Some("pcd_loader_service not available".to_string());
            log_warn!(
                LOGGER_NAME,
                "pcd_loader_service not available, skipping map update request"
            );
            return Ok(false);
        }

        // Check if a request is already pending
        if self.request_pending.swap(true, Ordering::SeqCst) {
            log_debug!(LOGGER_NAME, "Map update request already pending, skipping");
            return Ok(false);
        }

        // Build the request
        let request = GetDifferentialPointCloudMap_Request {
            area: AreaInfo {
                center_x: position.x as f32,
                center_y: position.y as f32,
                radius: map_radius,
            },
            cached_ids: self.map_module.get_loaded_tile_ids(),
        };

        log_debug!(
            LOGGER_NAME,
            "Requesting map at ({:.1}, {:.1}) radius={:.0}m, cached_tiles={}",
            request.area.center_x,
            request.area.center_y,
            request.area.radius,
            request.cached_ids.len()
        );

        // Clone what we need for the callback
        let map_module = Arc::clone(&self.map_module);
        let request_pending = Arc::clone(&self.request_pending);
        let status = Arc::clone(&self.status);

        // Send request with callback
        let _promise = self.client.call_then(request, move |response| {
            Self::handle_response(response, &map_module, &status);
            request_pending.store(false, Ordering::SeqCst);
        })?;

        Ok(true)
    }

    /// Handle response from the map service.
    fn handle_response(
        response: autoware_map_msgs::srv::GetDifferentialPointCloudMap_Response,
        map_module: &MapUpdateModule,
        status: &RwLock<MapLoaderStatus>,
    ) {
        let start_time = Instant::now();

        // Process tiles to add
        let mut tiles_added = 0;
        let mut points_added = 0;

        for cell in &response.new_pointcloud_with_ids {
            match Self::convert_cell_to_tile(cell) {
                Ok(tile) => {
                    points_added += tile.points.len();
                    map_module.add_tile(tile);
                    tiles_added += 1;
                }
                Err(e) => {
                    log_error!(
                        LOGGER_NAME,
                        "Failed to convert map cell '{}': {e}",
                        cell.cell_id
                    );
                }
            }
        }

        // Process tiles to remove
        let mut tiles_removed = 0;
        for id in &response.ids_to_remove {
            if map_module.remove_tile(id) {
                tiles_removed += 1;
            }
        }

        let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        // Update status
        {
            let mut s = status.write();
            s.last_request_success = true;
            s.tiles_added = tiles_added;
            s.tiles_removed = tiles_removed;
            s.points_added = points_added;
            s.last_update_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            s.error_message = None;
        }

        if tiles_added > 0 || tiles_removed > 0 {
            log_info!(
                LOGGER_NAME,
                "Map update: +{tiles_added} tiles ({points_added} points), -{tiles_removed} tiles, took {elapsed_ms:.1}ms"
            );
        }
    }

    /// Convert a PointCloudMapCellWithID to a MapTile.
    fn convert_cell_to_tile(cell: &PointCloudMapCellWithID) -> Result<MapTile, anyhow::Error> {
        // Convert PointCloud2 to Vec<[f32; 3]>
        let points = pointcloud::from_pointcloud2(&cell.pointcloud)?;

        // Calculate center from point cloud
        let center = if points.is_empty() {
            Point {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            }
        } else {
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

        Ok(MapTile {
            id: cell.cell_id.clone(),
            center,
            points,
        })
    }
}
