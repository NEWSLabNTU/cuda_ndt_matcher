use arc_swap::ArcSwap;
use autoware_internal_debug_msgs::msg::{Float32Stamped, Int32Stamped};
use autoware_internal_localization_msgs::srv::PoseWithCovarianceStamped as PoseWithCovSrv;
use geometry_msgs::msg::{PoseArray, PoseStamped, PoseWithCovarianceStamped};
use parking_lot::Mutex;
use rclrs::{Publisher, Service, Subscription};
use sensor_msgs::msg::PointCloud2;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32};
use std_srvs::srv::{SetBool, Trigger};
use tf2_msgs::msg::TFMessage;
use visualization_msgs::msg::MarkerArray;

use crate::alignment::DualNdtManager;
use crate::alignment::batch::ScanQueue;
use crate::io::diagnostics::DiagnosticsInterface;
use crate::io::params::NdtParams;
use crate::map::{DynamicMapLoader, MapUpdateModule};
use crate::transform::SmartPoseBuffer;
use crate::transform::tf_handler;

// Type aliases
pub(crate) type SetBoolRequest = std_srvs::srv::SetBool_Request;
pub(crate) type SetBoolResponse = std_srvs::srv::SetBool_Response;
pub(crate) type TriggerRequest = std_srvs::srv::Trigger_Request;
pub(crate) type TriggerResponse = std_srvs::srv::Trigger_Response;
pub(crate) type PoseWithCovSrvRequest =
    autoware_internal_localization_msgs::srv::PoseWithCovarianceStamped_Request;
pub(crate) type PoseWithCovSrvResponse =
    autoware_internal_localization_msgs::srv::PoseWithCovarianceStamped_Response;

pub(crate) const NODE_NAME: &str = "ndt_scan_matcher";

/// Holds debug and visualization publishers
#[derive(Clone)]
pub(crate) struct DebugPublishers {
    // TF broadcaster (publishes to /tf)
    pub(crate) tf_pub: Publisher<TFMessage>,

    // Visualization
    pub(crate) ndt_marker_pub: Publisher<MarkerArray>,
    pub(crate) points_aligned_pub: Publisher<PointCloud2>,
    pub(crate) monte_carlo_marker_pub: Publisher<MarkerArray>,

    // Debug metrics
    pub(crate) transform_probability_pub: Publisher<Float32Stamped>,
    pub(crate) nvtl_pub: Publisher<Float32Stamped>,
    pub(crate) iteration_num_pub: Publisher<Int32Stamped>,
    pub(crate) exe_time_pub: Publisher<Float32Stamped>,
    pub(crate) oscillation_count_pub: Publisher<Int32Stamped>,

    // Pose tracking
    pub(crate) initial_pose_cov_pub: Publisher<PoseWithCovarianceStamped>,
    pub(crate) initial_to_result_distance_pub: Publisher<Float32Stamped>,
    pub(crate) initial_to_result_distance_old_pub: Publisher<Float32Stamped>,
    pub(crate) initial_to_result_distance_new_pub: Publisher<Float32Stamped>,
    pub(crate) initial_to_result_relative_pose_pub: Publisher<PoseStamped>,

    // No-ground scoring (debug)
    pub(crate) no_ground_points_aligned_pub: Publisher<PointCloud2>,
    pub(crate) no_ground_transform_probability_pub: Publisher<Float32Stamped>,
    pub(crate) no_ground_nvtl_pub: Publisher<Float32Stamped>,

    // Per-point score visualization (voxel_score_points with RGB colors)
    #[cfg(feature = "debug-markers")]
    pub(crate) voxel_score_points_pub: Publisher<PointCloud2>,

    // MULTI_NDT covariance debug: poses from offset alignments
    pub(crate) multi_ndt_pose_pub: Publisher<PoseArray>,

    // MULTI_NDT covariance debug: initial poses before alignment
    pub(crate) multi_initial_pose_pub: Publisher<PoseArray>,

    // Debug map: currently loaded point cloud map
    pub(crate) debug_loaded_pointcloud_map_pub: Publisher<PointCloud2>,
}

/// Shared state passed to the `on_points` callback.
///
/// Groups the 18 parameters that were previously passed individually,
/// making the callback signature manageable and the cloning explicit.
#[derive(Clone)]
pub(crate) struct OnPointsContext {
    pub(crate) ndt_manager: Arc<DualNdtManager>,
    pub(crate) map_module: Arc<MapUpdateModule>,
    pub(crate) map_loader: Arc<DynamicMapLoader>,
    pub(crate) map_points: Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
    pub(crate) pose_buffer: Arc<SmartPoseBuffer>,
    pub(crate) latest_sensor_points: Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
    pub(crate) enabled: Arc<AtomicBool>,
    pub(crate) skip_counter: Arc<AtomicI32>,
    pub(crate) callback_count: Arc<AtomicI32>,
    pub(crate) align_count: Arc<AtomicI32>,
    pub(crate) pose_pub: Publisher<PoseStamped>,
    pub(crate) pose_cov_pub: Publisher<PoseWithCovarianceStamped>,
    pub(crate) debug_pubs: DebugPublishers,
    pub(crate) diagnostics: Arc<Mutex<DiagnosticsInterface>>,
    pub(crate) params: Arc<NdtParams>,
    pub(crate) tf_handler: Arc<tf_handler::TfHandler>,
    pub(crate) scan_queue: Option<Arc<ScanQueue>>,
}

// Note: Many fields appear "unused" but are actually used via cloned references
// passed to subscription/service callbacks. Rust's dead code analysis doesn't
// track usage through closures.
#[allow(dead_code)]
pub(crate) struct NdtScanMatcherNode {
    // Subscriptions (stored to keep alive)
    pub(super) _points_sub: Subscription<PointCloud2>,
    pub(super) _initial_pose_sub: Subscription<PoseWithCovarianceStamped>,
    pub(super) _regularization_pose_sub: Subscription<PoseWithCovarianceStamped>,
    pub(super) _map_sub: Subscription<PointCloud2>,

    // Publishers - Core pose output
    pub(super) pose_pub: Publisher<PoseStamped>,
    pub(super) pose_cov_pub: Publisher<PoseWithCovarianceStamped>,

    // Publishers - Debug and visualization
    pub(super) debug_pubs: DebugPublishers,

    // Diagnostics
    pub(super) diagnostics: Arc<Mutex<DiagnosticsInterface>>,

    // Services
    pub(super) _trigger_srv: Service<SetBool>,
    pub(super) _ndt_align_srv: Service<PoseWithCovSrv>,
    pub(super) _map_update_srv: Service<Trigger>,

    // State
    pub(super) ndt_manager: Arc<DualNdtManager>,
    pub(super) map_module: Arc<MapUpdateModule>,
    pub(super) map_loader: Arc<DynamicMapLoader>,
    pub(super) map_points: Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
    pub(super) pose_buffer: Arc<SmartPoseBuffer>,
    pub(super) latest_sensor_points: Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
    pub(super) enabled: Arc<AtomicBool>,
    pub(super) params: Arc<NdtParams>,

    // TF2 handler for sensor frame transforms
    pub(super) tf_handler: Arc<tf_handler::TfHandler>,

    // Scan queue for batch processing (optional, enabled via params.batch.enabled)
    // Wrapped in Arc so it can be shared with the subscription callback
    pub(super) scan_queue: Option<Arc<ScanQueue>>,
}
