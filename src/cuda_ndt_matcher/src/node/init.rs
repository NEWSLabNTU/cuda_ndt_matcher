use anyhow::Result;
use arc_swap::ArcSwap;
use geometry_msgs::msg::{PoseStamped, PoseWithCovariance, PoseWithCovarianceStamped};
use parking_lot::Mutex;
use rclrs::{
    Node, Publisher, QoSHistoryPolicy, QoSProfile, Service, Subscription, SubscriptionOptions,
    log_debug, log_error, log_info,
};
use sensor_msgs::msg::PointCloud2;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicI32, Ordering},
};
use std_srvs::srv::SetBool;

use super::{
    publishers, services,
    state::{
        DebugPublishers, NODE_NAME, NdtScanMatcherNode, OnPointsContext, SetBoolRequest,
        SetBoolResponse,
    },
};
use crate::{
    alignment::{
        DualNdtManager,
        batch::{ScanQueue, ScanQueueConfig, ScanResult},
    },
    io::{diagnostics::DiagnosticsInterface, params::NdtParams},
    map::{DynamicMapLoader, MapUpdateModule},
    transform::{SmartPoseBuffer, pose_utils, tf_handler},
};

impl NdtScanMatcherNode {
    pub(crate) fn new(node: &Node) -> Result<Self> {
        // Load parameters and initialize core components
        let params = Arc::new(NdtParams::from_node(node)?);
        log_info!(
            NODE_NAME,
            "NDT params: resolution={}, max_iter={}, epsilon={}",
            params.ndt.resolution,
            params.ndt.max_iterations,
            params.ndt.trans_epsilon
        );

        let ndt_manager = Arc::new(DualNdtManager::new((*params).clone())?);
        let tf_handler = tf_handler::TfHandler::new(node)?;
        log_info!(NODE_NAME, "TF handler initialized for sensor transforms");

        // Shared state
        let map_points: Arc<ArcSwap<Option<Vec<[f32; 3]>>>> = Arc::new(ArcSwap::from_pointee(None));
        let pose_buffer = Arc::new(SmartPoseBuffer::new(
            params.validation.initial_pose_timeout_sec,
            params.validation.initial_pose_distance_tolerance_m,
        ));
        let latest_sensor_points: Arc<ArcSwap<Option<Vec<[f32; 3]>>>> =
            Arc::new(ArcSwap::from_pointee(None));
        let enabled = Arc::new(AtomicBool::new(false));
        let skip_counter = Arc::new(AtomicI32::new(0));
        let callback_count = Arc::new(AtomicI32::new(0));
        let align_count = Arc::new(AtomicI32::new(0));

        // Map module
        let map_module = Arc::new(MapUpdateModule::new(params.dynamic_map.clone()));
        log_info!(
            NODE_NAME,
            "Map module: update_distance={}, map_radius={}, lidar_radius={}",
            params.dynamic_map.update_distance,
            params.dynamic_map.map_radius,
            params.dynamic_map.lidar_radius
        );
        let map_loader = Arc::new(DynamicMapLoader::new(
            node,
            "pcd_loader_service",
            Arc::clone(&map_module),
        )?);

        // Create publishers, diagnostics, batch queue
        let (pose_pub, pose_cov_pub, debug_pubs) = create_publishers(node)?;
        let diagnostics = Arc::new(Mutex::new(DiagnosticsInterface::new(node)?));
        let scan_queue =
            create_batch_queue(&params, &ndt_manager, &pose_pub, &pose_cov_pub, &debug_pubs);

        // Create subscriptions
        let (points_sub, initial_pose_sub, regularization_pose_sub, map_sub) =
            create_subscriptions(
                node,
                &ndt_manager,
                &map_module,
                &map_loader,
                &map_points,
                &pose_buffer,
                &latest_sensor_points,
                &enabled,
                &skip_counter,
                &callback_count,
                &align_count,
                &pose_pub,
                &pose_cov_pub,
                &debug_pubs,
                &diagnostics,
                &params,
                &tf_handler,
                &scan_queue,
            )?;

        // Create services
        let (trigger_srv, ndt_align_srv, map_update_srv) = create_services(
            node,
            &ndt_manager,
            &map_module,
            &map_points,
            &latest_sensor_points,
            &pose_buffer,
            &enabled,
            &params,
            &debug_pubs,
        )?;

        initialize_debug_output();

        log_info!(NODE_NAME, "NDT scan matcher node initialized");

        Ok(Self {
            _points_sub: points_sub,
            _initial_pose_sub: initial_pose_sub,
            _regularization_pose_sub: regularization_pose_sub,
            _map_sub: map_sub,
            pose_pub,
            pose_cov_pub,
            debug_pubs,
            diagnostics,
            _trigger_srv: trigger_srv,
            _ndt_align_srv: ndt_align_srv,
            _map_update_srv: map_update_srv,
            ndt_manager,
            map_module,
            map_loader,
            map_points,
            pose_buffer,
            latest_sensor_points,
            enabled,
            params,
            tf_handler,
            scan_queue,
        })
    }
}

/// Create all publishers (core pose output + debug/visualization).
fn create_publishers(
    node: &Node,
) -> Result<(
    Publisher<PoseStamped>,
    Publisher<PoseWithCovarianceStamped>,
    DebugPublishers,
)> {
    let pose_pub = node.create_publisher("pose")?;
    let pose_cov_pub = node.create_publisher("pose_with_covariance")?;

    let debug_pubs = DebugPublishers {
        tf_pub: node.create_publisher("/tf")?,
        ndt_marker_pub: node.create_publisher("ndt_marker")?,
        points_aligned_pub: node.create_publisher("points_aligned")?,
        monte_carlo_marker_pub: node.create_publisher("monte_carlo_initial_pose_marker")?,
        transform_probability_pub: node.create_publisher("transform_probability")?,
        nvtl_pub: node.create_publisher("nearest_voxel_transformation_likelihood")?,
        iteration_num_pub: node.create_publisher("iteration_num")?,
        exe_time_pub: node.create_publisher("exe_time_ms")?,
        oscillation_count_pub: node.create_publisher("local_optimal_solution_oscillation_num")?,
        initial_pose_cov_pub: node.create_publisher("initial_pose_with_covariance")?,
        initial_to_result_distance_pub: node.create_publisher("initial_to_result_distance")?,
        initial_to_result_distance_old_pub: node
            .create_publisher("initial_to_result_distance_old")?,
        initial_to_result_distance_new_pub: node
            .create_publisher("initial_to_result_distance_new")?,
        initial_to_result_relative_pose_pub: node
            .create_publisher("initial_to_result_relative_pose")?,
        no_ground_points_aligned_pub: node.create_publisher("points_aligned_no_ground")?,
        no_ground_transform_probability_pub: node
            .create_publisher("no_ground_transform_probability")?,
        no_ground_nvtl_pub: node
            .create_publisher("no_ground_nearest_voxel_transformation_likelihood")?,
        #[cfg(feature = "debug-markers")]
        voxel_score_points_pub: node.create_publisher("voxel_score_points")?,
        multi_ndt_pose_pub: node.create_publisher("multi_ndt_pose")?,
        multi_initial_pose_pub: node.create_publisher("multi_initial_pose")?,
        debug_loaded_pointcloud_map_pub: node.create_publisher("debug/loaded_pointcloud_map")?,
    };

    Ok((pose_pub, pose_cov_pub, debug_pubs))
}

/// Create scan queue for batch GPU processing (if enabled in params).
fn create_batch_queue(
    params: &Arc<NdtParams>,
    ndt_manager: &Arc<DualNdtManager>,
    pose_pub: &Publisher<PoseStamped>,
    pose_cov_pub: &Publisher<PoseWithCovarianceStamped>,
    debug_pubs: &DebugPublishers,
) -> Option<Arc<ScanQueue>> {
    if !params.batch.enabled {
        return None;
    }

    log_info!(
        NODE_NAME,
        "Batch processing enabled: trigger={}, timeout={}ms, max_depth={}",
        params.batch.batch_trigger,
        params.batch.timeout_ms,
        params.batch.max_queue_depth
    );

    let config = ScanQueueConfig::from_params(&params.batch);

    // Alignment function: locks NDT manager and runs batch alignment
    let align_ndt_manager = Arc::clone(ndt_manager);
    let align_fn: crate::alignment::batch::AlignFn = Arc::new(move |requests| {
        let manager = align_ndt_manager.lock();
        let results = manager.align_batch_scans(requests)?;
        Ok(results)
    });

    // Result callback: publishes converged poses
    let result_pose_pub = pose_pub.clone();
    let result_pose_cov_pub = pose_cov_pub.clone();
    let result_debug_pubs = debug_pubs.clone();
    let result_params = Arc::clone(params);
    let result_callback: crate::alignment::batch::ResultCallback =
        Arc::new(move |results: Vec<ScanResult>| {
            for result in results {
                if !result.converged {
                    log_debug!(
                        NODE_NAME,
                        "Batch result skipped (not converged): ts_ns={}, score={:.3}",
                        result.timestamp_ns,
                        result.score
                    );
                    continue;
                }

                let pose = pose_utils::pose_from_isometry(&result.pose);

                let pose_msg = PoseStamped {
                    header: result.header.clone(),
                    pose: pose.clone(),
                };
                if let Err(e) = result_pose_pub.publish(&pose_msg) {
                    log_error!(NODE_NAME, "Failed to publish batch pose: {e}");
                }

                let pose_cov_msg = PoseWithCovarianceStamped {
                    header: result.header.clone(),
                    pose: PoseWithCovariance {
                        pose: pose.clone(),
                        covariance: result_params.covariance.output_pose_covariance,
                    },
                };
                if let Err(e) = result_pose_cov_pub.publish(&pose_cov_msg) {
                    log_error!(
                        NODE_NAME,
                        "Failed to publish batch pose with covariance: {e}"
                    );
                }

                publishers::publish_tf(
                    &result_debug_pubs.tf_pub,
                    &result.timestamp,
                    &pose,
                    &result_params.frame.map_frame,
                    &result_params.frame.ndt_base_frame,
                );

                log_debug!(
                    NODE_NAME,
                    "Batch result published: ts_ns={}, iter={}, score={:.3}, latency={:.1}ms",
                    result.timestamp_ns,
                    result.iterations,
                    result.score,
                    result.latency_ms
                );
            }
        });

    Some(Arc::new(ScanQueue::new(config, align_fn, result_callback)))
}

/// Create all 4 subscriptions (points_raw, EKF pose, regularization, map).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn create_subscriptions(
    node: &Node,
    ndt_manager: &Arc<DualNdtManager>,
    map_module: &Arc<MapUpdateModule>,
    map_loader: &Arc<DynamicMapLoader>,
    map_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
    pose_buffer: &Arc<SmartPoseBuffer>,
    latest_sensor_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
    enabled: &Arc<AtomicBool>,
    skip_counter: &Arc<AtomicI32>,
    callback_count: &Arc<AtomicI32>,
    align_count: &Arc<AtomicI32>,
    pose_pub: &Publisher<PoseStamped>,
    pose_cov_pub: &Publisher<PoseWithCovarianceStamped>,
    debug_pubs: &DebugPublishers,
    diagnostics: &Arc<Mutex<DiagnosticsInterface>>,
    params: &Arc<NdtParams>,
    tf_handler: &Arc<tf_handler::TfHandler>,
    scan_queue: &Option<Arc<ScanQueue>>,
) -> Result<(
    Subscription<PointCloud2>,
    Subscription<PoseWithCovarianceStamped>,
    Subscription<PoseWithCovarianceStamped>,
    Subscription<PointCloud2>,
)> {
    let sensor_qos = QoSProfile {
        history: QoSHistoryPolicy::KeepLast { depth: 1 },
        ..QoSProfile::sensor_data_default()
    };

    // Points subscription with OnPointsContext
    let points_sub = {
        let ctx = OnPointsContext {
            ndt_manager: Arc::clone(ndt_manager),
            map_module: Arc::clone(map_module),
            map_loader: Arc::clone(map_loader),
            map_points: Arc::clone(map_points),
            pose_buffer: Arc::clone(pose_buffer),
            latest_sensor_points: Arc::clone(latest_sensor_points),
            enabled: Arc::clone(enabled),
            skip_counter: Arc::clone(skip_counter),
            callback_count: Arc::clone(callback_count),
            align_count: Arc::clone(align_count),
            pose_pub: pose_pub.clone(),
            pose_cov_pub: pose_cov_pub.clone(),
            debug_pubs: debug_pubs.clone(),
            diagnostics: Arc::clone(diagnostics),
            params: Arc::clone(params),
            tf_handler: Arc::clone(tf_handler),
            scan_queue: scan_queue.clone(),
        };

        let mut opts = SubscriptionOptions::new("points_raw");
        opts.qos = sensor_qos;

        node.create_subscription(opts, move |msg: PointCloud2| {
            NdtScanMatcherNode::on_points(msg, &ctx);
        })?
    };

    // EKF pose subscription (depth 100 to buffer during init)
    let initial_pose_sub = {
        let pose_buffer = Arc::clone(pose_buffer);

        let mut opts = SubscriptionOptions::new("ekf_pose_with_covariance");
        opts.qos = QoSProfile {
            history: QoSHistoryPolicy::KeepLast { depth: 100 },
            ..QoSProfile::default()
        };

        node.create_subscription(opts, move |msg: PoseWithCovarianceStamped| {
            #[cfg(feature = "debug-output")]
            {
                let p = &msg.pose.pose.position;
                let q = &msg.pose.pose.orientation;
                let ts = &msg.header.stamp;
                log_info!(
                    NODE_NAME,
                    "[EKF_IN] ts={}.{:09} pos=({:.3}, {:.3}, {:.3}) quat=({:.6}, {:.6}, {:.6}, {:.6})",
                    ts.sec, ts.nanosec,
                    p.x, p.y, p.z,
                    q.x, q.y, q.z, q.w
                );
            }
            pose_buffer.push_back(msg);
        })?
    };

    log_debug!(
        NODE_NAME,
        "EKF pose subscription topic (after remapping): {}",
        initial_pose_sub.topic_name()
    );

    // Regularization pose subscription (GNSS)
    let regularization_pose_sub = {
        let ndt_manager = Arc::clone(ndt_manager);
        let params = Arc::clone(params);

        let mut opts = SubscriptionOptions::new("regularization_pose_with_covariance");
        opts.qos = sensor_qos;

        node.create_subscription(opts, move |msg: PoseWithCovarianceStamped| {
            if !params.regularization.enabled {
                return;
            }
            ndt_manager.set_regularization_pose(&msg.pose.pose);
        })?
    };

    // Map subscription
    let map_sub = {
        let map_module = Arc::clone(map_module);
        let map_points = Arc::clone(map_points);
        let ndt_manager = Arc::clone(ndt_manager);
        let debug_pubs = debug_pubs.clone();
        let params = Arc::clone(params);

        let mut opts = SubscriptionOptions::new("pointcloud_map");
        opts.qos = QoSProfile::default();

        node.create_subscription(opts, move |msg: PointCloud2| {
            services::on_map_received(
                msg,
                &map_module,
                &map_points,
                &ndt_manager,
                &debug_pubs,
                &params,
            );
        })?
    };

    Ok((
        points_sub,
        initial_pose_sub,
        regularization_pose_sub,
        map_sub,
    ))
}

/// Create all 3 services (trigger, ndt_align, map_update).
#[allow(clippy::too_many_arguments)]
fn create_services(
    node: &Node,
    ndt_manager: &Arc<DualNdtManager>,
    map_module: &Arc<MapUpdateModule>,
    map_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
    latest_sensor_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
    pose_buffer: &Arc<SmartPoseBuffer>,
    enabled: &Arc<AtomicBool>,
    params: &Arc<NdtParams>,
    debug_pubs: &DebugPublishers,
) -> Result<(
    Service<SetBool>,
    Service<autoware_internal_localization_msgs::srv::PoseWithCovarianceStamped>,
    Service<std_srvs::srv::Trigger>,
)> {
    // Debug: Track time from init request to tracking enabled
    #[cfg(feature = "debug-output")]
    let init_request_time: Arc<std::sync::Mutex<Option<std::time::Instant>>> =
        Arc::new(std::sync::Mutex::new(None));
    #[cfg(feature = "debug-output")]
    let init_request_time_for_align = Arc::clone(&init_request_time);
    #[cfg(feature = "debug-output")]
    let init_request_time_for_trigger = Arc::clone(&init_request_time);

    // Trigger service (enable/disable NDT)
    let trigger_srv = {
        let enabled = Arc::clone(enabled);
        let pose_buffer = Arc::clone(pose_buffer);

        node.create_service::<SetBool, _>(
            "trigger_node_srv",
            move |req: SetBoolRequest, _info: rclrs::ServiceInfo| {
                enabled.store(req.data, Ordering::SeqCst);
                if req.data {
                    #[cfg(feature = "debug-output")]
                    {
                        if let Some(start) = init_request_time_for_trigger.lock().unwrap().take() {
                            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                            log_info!(NODE_NAME, "Init-to-tracking time: {:.2}ms", elapsed_ms);
                            crate::io::debug_writer::write_init_to_tracking(elapsed_ms);
                        }
                    }
                    pose_buffer.clear();
                    log_info!(NODE_NAME, "NDT scan matcher enabled (pose buffer cleared)");
                } else {
                    log_info!(NODE_NAME, "NDT scan matcher disabled");
                }
                SetBoolResponse {
                    success: true,
                    message: format!(
                        "NDT scan matcher {}",
                        if req.data { "enabled" } else { "disabled" }
                    ),
                }
            },
        )?
    };

    // NDT align service (initial pose estimation)
    let ndt_align_srv = {
        let ndt_manager = Arc::clone(ndt_manager);
        let map_points = Arc::clone(map_points);
        let latest_sensor_points = Arc::clone(latest_sensor_points);
        let params = Arc::clone(params);
        let monte_carlo_pub = debug_pubs.monte_carlo_marker_pub.clone();

        node.create_service::<autoware_internal_localization_msgs::srv::PoseWithCovarianceStamped, _>(
            "ndt_align_srv",
            move |req: super::state::PoseWithCovSrvRequest, _info: rclrs::ServiceInfo| {
                #[cfg(feature = "debug-output")]
                {
                    let mut guard = init_request_time_for_align.lock().unwrap();
                    if guard.is_none() {
                        *guard = Some(std::time::Instant::now());
                    }
                }

                services::on_ndt_align(
                    req,
                    &ndt_manager,
                    &map_points,
                    &latest_sensor_points,
                    &params,
                    &monte_carlo_pub,
                )
            },
        )?
    };

    // Map update service
    let map_update_srv = {
        let map_module = Arc::clone(map_module);
        let map_points = Arc::clone(map_points);
        let ndt_manager = Arc::clone(ndt_manager);
        let pose_buffer = Arc::clone(pose_buffer);

        node.create_service::<std_srvs::srv::Trigger, _>(
            "map_update_srv",
            move |_req: super::state::TriggerRequest, _info: rclrs::ServiceInfo| {
                services::on_map_update(&map_module, &map_points, &ndt_manager, &pose_buffer)
            },
        )?
    };

    Ok((trigger_srv, ndt_align_srv, map_update_srv))
}

/// Clear debug output file at startup (only with debug-output feature).
fn initialize_debug_output() {
    #[cfg(feature = "debug-output")]
    {
        crate::io::debug_writer::clear_debug_file();
        log_info!(
            NODE_NAME,
            "Debug output cleared: {}",
            crate::io::debug_writer::debug_file_path()
        );
    }
}
