use arc_swap::ArcSwap;
use geometry_msgs::msg::{PoseWithCovariance, PoseWithCovarianceStamped};
use rclrs::{log_debug, log_error, log_info, log_warn, Publisher};
use sensor_msgs::msg::PointCloud2;
use std::sync::Arc;
use std_msgs::msg::Header;
use visualization_msgs::msg::MarkerArray;

use super::state::{
    DebugPublishers, NdtScanMatcherNode, PoseWithCovSrvRequest, PoseWithCovSrvResponse,
    TriggerResponse, NODE_NAME,
};
use crate::dual_ndt_manager::DualNdtManager;
use crate::initial_pose;
use crate::map_module::MapUpdateModule;
use crate::params::NdtParams;
use crate::pointcloud;
use crate::pose_buffer::SmartPoseBuffer;
use crate::visualization::{self, ParticleMarkerConfig};

/// Handle NDT align service request
/// This service is called by pose_initializer with an initial pose guess.
/// It performs multi-particle NDT alignment using TPE and returns the best aligned pose.
/// This matches Autoware's behavior of sampling multiple initial poses to find the best match.
pub(crate) fn on_ndt_align(
    req: PoseWithCovSrvRequest,
    ndt_manager: &Arc<DualNdtManager>,
    map_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
    latest_sensor_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
    params: &NdtParams,
    monte_carlo_pub: &Publisher<MarkerArray>,
) -> PoseWithCovSrvResponse {
    log_info!(NODE_NAME, "NDT align service called");

    // Get initial pose from request
    let initial_pose = req.pose_with_covariance;

    // Get map points
    let map = map_points.load();
    let map = match map.as_ref() {
        Some(m) => m,
        None => {
            log_error!(NODE_NAME, "NDT align failed: No map loaded");
            return PoseWithCovSrvResponse {
                success: false,
                reliable: false,
                pose_with_covariance: initial_pose,
            };
        }
    };

    // Get sensor points
    let sensor_points = latest_sensor_points.load();
    let sensor_points = match sensor_points.as_ref() {
        Some(p) => p,
        None => {
            log_error!(NODE_NAME, "NDT align failed: No sensor points available");
            return PoseWithCovSrvResponse {
                success: false,
                reliable: false,
                pose_with_covariance: initial_pose,
            };
        }
    };

    // Run multi-particle initial pose estimation using TPE
    // Lock the active manager for the duration of pose estimation
    let mut manager = ndt_manager.lock();
    let score_threshold = params
        .score
        .converged_param_nearest_voxel_transformation_likelihood;

    let result = match initial_pose::estimate_initial_pose(
        &initial_pose,
        &mut manager,
        sensor_points,
        map,
        &params.initial_pose,
        params.ndt.resolution,
        score_threshold,
    ) {
        Ok(r) => r,
        Err(e) => {
            log_error!(NODE_NAME, "Initial pose estimation failed: {e}");
            return PoseWithCovSrvResponse {
                success: false,
                reliable: false,
                pose_with_covariance: initial_pose,
            };
        }
    };

    log_info!(
        NODE_NAME,
        "NDT align complete: converged=true, score={:.3}, reliable={}, particles={}",
        result.score,
        result.reliable,
        result.particles.len()
    );

    // Publish Monte Carlo particle visualization with multiple color schemes
    let markers = visualization::create_monte_carlo_markers_enhanced(
        &initial_pose.header,
        &result.particles,
        result.score,
        &ParticleMarkerConfig::default(),
    );
    if let Err(e) = monte_carlo_pub.publish(&markers) {
        log_debug!(NODE_NAME, "Failed to publish Monte Carlo markers: {e}");
    }

    // Build result with best aligned pose
    let result_pose = PoseWithCovarianceStamped {
        header: initial_pose.header,
        pose: PoseWithCovariance {
            pose: result.pose_with_covariance.pose.pose,
            covariance: params.covariance.output_pose_covariance,
        },
    };

    PoseWithCovSrvResponse {
        success: true,
        reliable: result.reliable,
        pose_with_covariance: result_pose,
    }
}

/// Handle map point cloud received
pub(crate) fn on_map_received(
    msg: PointCloud2,
    map_module: &Arc<MapUpdateModule>,
    map_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
    ndt_manager: &Arc<DualNdtManager>,
    debug_pubs: &DebugPublishers,
    params: &NdtParams,
) {
    // Convert point cloud
    let points = match pointcloud::from_pointcloud2(&msg) {
        Ok(pts) => pts,
        Err(e) => {
            log_error!(NODE_NAME, "Failed to convert map point cloud: {e}");
            return;
        }
    };

    log_info!(NODE_NAME, "Received map with {} points", points.len());

    // Load into map module
    map_module.load_full_map(points.clone());

    // Update shared map points
    map_points.store(Arc::new(Some(points.clone())));

    // Publish debug map for visualization
    let debug_map_msg = pointcloud::to_pointcloud2(
        &points,
        &Header {
            stamp: msg.header.stamp.clone(),
            frame_id: params.frame.map_frame.clone(),
        },
    );
    let _ = debug_pubs
        .debug_loaded_pointcloud_map_pub
        .publish(&debug_map_msg);

    // Update NDT target (blocking for initial map load)
    if let Err(e) = ndt_manager.set_target(&points) {
        log_error!(NODE_NAME, "Failed to set NDT target: {e}");
    } else {
        log_info!(NODE_NAME, "NDT target updated with map");
    }
}

/// Handle map update service request
pub(crate) fn on_map_update(
    map_module: &Arc<MapUpdateModule>,
    map_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
    ndt_manager: &Arc<DualNdtManager>,
    pose_buffer: &Arc<SmartPoseBuffer>,
) -> TriggerResponse {
    // Get current position from latest pose in buffer
    let position = match pose_buffer.latest() {
        Some(p) => geometry_msgs::msg::Point {
            x: p.pose.pose.position.x,
            y: p.pose.pose.position.y,
            z: p.pose.pose.position.z,
        },
        None => {
            return TriggerResponse {
                success: false,
                message: "No position available for map update".to_string(),
            };
        }
    };

    // Check if update is needed
    let should_update = map_module.should_update(&position);
    let out_of_range = map_module.out_of_map_range(&position);

    if out_of_range {
        log_warn!(
            NODE_NAME,
            "Position is out of map range - may need new map data"
        );
    }

    // Perform map update
    let result = map_module.update_map(&position);

    if result.updated {
        log_info!(
            NODE_NAME,
            "Map updated: {} tiles, {} points, distance={:.1}m",
            result.tiles_loaded,
            result.total_points,
            result.distance_from_last_update
        );

        // Update shared map points with filtered map
        if let Some(filtered_points) = map_module.get_map_points() {
            map_points.store(Arc::new(Some(filtered_points.clone())));

            // Start non-blocking NDT target update
            let started = ndt_manager.start_background_update(filtered_points);
            log_debug!(
                NODE_NAME,
                "Map update service: background NDT update started={started}"
            );
        }
    }

    TriggerResponse {
        success: true,
        message: format!(
            "updated={}, tiles={}, points={}, distance={:.1}m, should_update={}, out_of_range={}",
            result.updated,
            result.tiles_loaded,
            result.total_points,
            result.distance_from_last_update,
            should_update,
            out_of_range
        ),
    }
}

impl NdtScanMatcherNode {
    /// Load map from points (called externally or via service)
    #[allow(dead_code)]
    pub(crate) fn set_map(&self, points: Vec<[f32; 3]>) {
        log_info!(NODE_NAME, "Loading map with {} points", points.len());
        self.map_points.store(Arc::new(Some(points.clone())));

        // Blocking set for initial map load
        if let Err(e) = self.ndt_manager.set_target(&points) {
            log_error!(NODE_NAME, "Failed to set NDT target: {e}");
        }
    }
}
