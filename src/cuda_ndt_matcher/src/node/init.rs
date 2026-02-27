use anyhow::Result;
use arc_swap::ArcSwap;
use geometry_msgs::msg::{PoseWithCovariance, PoseWithCovarianceStamped};
use rclrs::{
    log_debug, log_error, log_info, Node, QoSHistoryPolicy, QoSProfile, SubscriptionOptions,
};
use sensor_msgs::msg::PointCloud2;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Arc;

use super::state::{
    DebugPublishers, NdtScanMatcherNode, OnPointsContext, SetBoolRequest, SetBoolResponse,
    NODE_NAME,
};
use super::{publishers, services};
use crate::alignment::batch::{ScanQueue, ScanQueueConfig, ScanResult};
use crate::alignment::DualNdtManager;
use crate::io::diagnostics::DiagnosticsInterface;
use crate::io::params::NdtParams;
use crate::map::{DynamicMapLoader, MapUpdateModule};
use crate::transform::pose_utils;
use crate::transform::tf_handler;
use crate::transform::SmartPoseBuffer;
use geometry_msgs::msg::PoseStamped;
use parking_lot::Mutex;
use std_srvs::srv::SetBool;

impl NdtScanMatcherNode {
    pub(crate) fn new(node: &Node) -> Result<Self> {
        // Load parameters
        let params = Arc::new(NdtParams::from_node(node)?);
        log_info!(
            NODE_NAME,
            "NDT params: resolution={}, max_iter={}, epsilon={}",
            params.ndt.resolution,
            params.ndt.max_iterations,
            params.ndt.trans_epsilon
        );

        // Initialize NDT manager (dual for non-blocking updates)
        let ndt_manager = Arc::new(DualNdtManager::new((*params).clone())?);

        // Initialize TF2 handler for sensor frame transforms
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
        // Start disabled - wait for trigger_node service to enable
        // This matches Autoware's behavior: pose_initializer refines the initial pose
        // via ndt_align_srv, sends it to EKF, then enables NDT
        let enabled = Arc::new(AtomicBool::new(false));
        // Track consecutive skips due to low score (like Autoware's skipping_publish_num)
        let skip_counter = Arc::new(AtomicI32::new(0));

        // Debug counters for callback tracking
        let callback_count = Arc::new(AtomicI32::new(0));
        let align_count = Arc::new(AtomicI32::new(0));

        // Note: We rely on QoS KeepLast(1) to prevent duplicate message processing,
        // matching Autoware's approach. No explicit timestamp deduplication needed.

        // Initialize map update module
        let map_module = Arc::new(MapUpdateModule::new(params.dynamic_map.clone()));
        log_info!(
            NODE_NAME,
            "Map module: update_distance={}, map_radius={}, lidar_radius={}",
            params.dynamic_map.update_distance,
            params.dynamic_map.map_radius,
            params.dynamic_map.lidar_radius
        );

        // Initialize dynamic map loader (service client for pcd_loader_service)
        let map_loader = Arc::new(DynamicMapLoader::new(
            node,
            "pcd_loader_service",
            Arc::clone(&map_module),
        )?);

        // QoS for sensor data
        let sensor_qos = QoSProfile {
            history: QoSHistoryPolicy::KeepLast { depth: 1 },
            ..QoSProfile::sensor_data_default()
        };

        // Publishers - Core pose output
        // Use actual topic names directly since rclrs doesn't support launch file remappings
        let pose_pub = node.create_publisher("pose")?;
        let pose_cov_pub = node.create_publisher("pose_with_covariance")?;

        // Publishers - Debug and visualization
        let debug_pubs = DebugPublishers {
            // TF broadcaster - publishes to /tf (absolute topic name)
            tf_pub: node.create_publisher("/tf")?,
            ndt_marker_pub: node.create_publisher("ndt_marker")?,
            points_aligned_pub: node.create_publisher("points_aligned")?,
            monte_carlo_marker_pub: node.create_publisher("monte_carlo_initial_pose_marker")?,
            transform_probability_pub: node.create_publisher("transform_probability")?,
            nvtl_pub: node.create_publisher("nearest_voxel_transformation_likelihood")?,
            iteration_num_pub: node.create_publisher("iteration_num")?,
            exe_time_pub: node.create_publisher("exe_time_ms")?,
            oscillation_count_pub: node
                .create_publisher("local_optimal_solution_oscillation_num")?,
            initial_pose_cov_pub: node.create_publisher("initial_pose_with_covariance")?,
            initial_to_result_distance_pub: node.create_publisher("initial_to_result_distance")?,
            initial_to_result_distance_old_pub: node
                .create_publisher("initial_to_result_distance_old")?,
            initial_to_result_distance_new_pub: node
                .create_publisher("initial_to_result_distance_new")?,
            initial_to_result_relative_pose_pub: node
                .create_publisher("initial_to_result_relative_pose")?,
            // No-ground scoring debug
            no_ground_points_aligned_pub: node.create_publisher("points_aligned_no_ground")?,
            no_ground_transform_probability_pub: node
                .create_publisher("no_ground_transform_probability")?,
            no_ground_nvtl_pub: node
                .create_publisher("no_ground_nearest_voxel_transformation_likelihood")?,
            // Per-point score visualization
            #[cfg(feature = "debug-markers")]
            voxel_score_points_pub: node.create_publisher("voxel_score_points")?,
            // MULTI_NDT covariance debug
            multi_ndt_pose_pub: node.create_publisher("multi_ndt_pose")?,
            multi_initial_pose_pub: node.create_publisher("multi_initial_pose")?,
            // Debug loaded map
            debug_loaded_pointcloud_map_pub: node
                .create_publisher("debug/loaded_pointcloud_map")?,
        };

        // Create diagnostics interface
        let diagnostics = Arc::new(Mutex::new(DiagnosticsInterface::new(node)?));

        // Initialize scan queue for batch processing (if enabled)
        // Must be created before the subscription so we can pass it to the callback
        let scan_queue: Option<Arc<ScanQueue>> = if params.batch.enabled {
            log_info!(
                NODE_NAME,
                "Batch processing enabled: trigger={}, timeout={}ms, max_depth={}",
                params.batch.batch_trigger,
                params.batch.timeout_ms,
                params.batch.max_queue_depth
            );

            let config = ScanQueueConfig::from_params(&params.batch);

            // Create alignment function that uses the NDT manager
            let align_ndt_manager = Arc::clone(&ndt_manager);
            let align_fn: crate::alignment::batch::AlignFn = Arc::new(move |requests| {
                // Get lock on active NDT manager
                let manager = align_ndt_manager.lock();
                // Use batch alignment through the ndt_cuda API
                let results = manager.align_batch_scans(requests)?;
                Ok(results)
            });

            // Create result callback that publishes poses
            let result_pose_pub = pose_pub.clone();
            let result_pose_cov_pub = pose_cov_pub.clone();
            let result_debug_pubs = debug_pubs.clone();
            let result_params = Arc::clone(&params);
            let result_callback: crate::alignment::batch::ResultCallback =
                Arc::new(move |results: Vec<ScanResult>| {
                    for result in results {
                        // Only publish if converged
                        if !result.converged {
                            log_debug!(
                                NODE_NAME,
                                "Batch result skipped (not converged): ts_ns={}, score={:.3}",
                                result.timestamp_ns,
                                result.score
                            );
                            continue;
                        }

                        // Convert Isometry3 to Pose
                        let pose = pose_utils::pose_from_isometry(&result.pose);

                        // Publish PoseStamped
                        let pose_msg = PoseStamped {
                            header: result.header.clone(),
                            pose: pose.clone(),
                        };
                        if let Err(e) = result_pose_pub.publish(&pose_msg) {
                            log_error!(NODE_NAME, "Failed to publish batch pose: {e}");
                        }

                        // Publish PoseWithCovarianceStamped with fixed covariance
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

                        // Publish TF transform
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
        } else {
            None
        };

        // Points subscription
        // Uses QoS KeepLast(1) to ensure we only process the latest message,
        // matching Autoware's approach (no explicit timestamp deduplication needed)
        let points_sub = {
            let ctx = OnPointsContext {
                ndt_manager: Arc::clone(&ndt_manager),
                map_module: Arc::clone(&map_module),
                map_loader: Arc::clone(&map_loader),
                map_points: Arc::clone(&map_points),
                pose_buffer: Arc::clone(&pose_buffer),
                latest_sensor_points: Arc::clone(&latest_sensor_points),
                enabled: Arc::clone(&enabled),
                skip_counter: Arc::clone(&skip_counter),
                callback_count: Arc::clone(&callback_count),
                align_count: Arc::clone(&align_count),
                pose_pub: pose_pub.clone(),
                pose_cov_pub: pose_cov_pub.clone(),
                debug_pubs: debug_pubs.clone(),
                diagnostics: Arc::clone(&diagnostics),
                params: Arc::clone(&params),
                tf_handler: Arc::clone(&tf_handler),
                scan_queue: scan_queue.clone(),
            };

            let mut opts = SubscriptionOptions::new("points_raw");
            opts.qos = sensor_qos;

            node.create_subscription(opts, move |msg: PointCloud2| {
                Self::on_points(msg, &ctx);
            })?
        };

        // Initial pose subscription - pushes to pose buffer for interpolation
        // Uses QoS depth 100 (matching Autoware) to buffer messages during node initialization.
        // This prevents losing early EKF messages before spin() starts processing callbacks.
        let initial_pose_sub = {
            let pose_buffer = Arc::clone(&pose_buffer);

            // Use relative topic name - remapping should be handled by rcl layer
            // Launch file remaps: ekf_pose_with_covariance -> /localization/pose_twist_fusion_filter/biased_pose_with_covariance
            let mut opts = SubscriptionOptions::new("ekf_pose_with_covariance");
            // QoS depth 100 matches Autoware's ndt_scan_matcher_core.cpp line 118
            opts.qos = QoSProfile {
                history: QoSHistoryPolicy::KeepLast { depth: 100 },
                ..QoSProfile::default()
            };

            node.create_subscription(opts, move |msg: PoseWithCovarianceStamped| {
                // Debug: log received EKF pose with timestamp (only with debug-output feature)
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

        // Log the actual topic name after remapping
        log_debug!(
            NODE_NAME,
            "EKF pose subscription topic (after remapping): {}",
            initial_pose_sub.topic_name()
        );

        // Regularization pose subscription (GNSS pose for regularization)
        let regularization_pose_sub = {
            let ndt_manager = Arc::clone(&ndt_manager);
            let params = Arc::clone(&params);

            let mut opts = SubscriptionOptions::new("regularization_pose_with_covariance");
            opts.qos = sensor_qos;

            node.create_subscription(opts, move |msg: PoseWithCovarianceStamped| {
                // Only process if regularization is enabled
                if !params.regularization.enabled {
                    return;
                }

                // Set the regularization reference pose in the NDT matcher
                ndt_manager.set_regularization_pose(&msg.pose.pose);
            })?
        };

        // Map subscription (for receiving point cloud map data)
        let map_sub = {
            let map_module = Arc::clone(&map_module);
            let map_points = Arc::clone(&map_points);
            let ndt_manager = Arc::clone(&ndt_manager);
            let debug_pubs = debug_pubs.clone();
            let params = Arc::clone(&params);

            let mut opts = SubscriptionOptions::new("pointcloud_map");
            opts.qos = QoSProfile::default(); // Reliable for map data

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

        // Debug: Track time from init request to tracking enabled
        #[cfg(feature = "debug-output")]
        let init_request_time: Arc<std::sync::Mutex<Option<std::time::Instant>>> =
            Arc::new(std::sync::Mutex::new(None));
        #[cfg(feature = "debug-output")]
        let init_request_time_for_align = Arc::clone(&init_request_time);
        #[cfg(feature = "debug-output")]
        let init_request_time_for_trigger = Arc::clone(&init_request_time);

        // Trigger service
        let trigger_srv = {
            let enabled = Arc::clone(&enabled);
            let pose_buffer = Arc::clone(&pose_buffer);

            node.create_service::<SetBool, _>(
                "trigger_node_srv",
                move |req: SetBoolRequest, _info: rclrs::ServiceInfo| {
                    enabled.store(req.data, Ordering::SeqCst);
                    // Clear pose buffer when enabling (matches Autoware behavior)
                    // This ensures we start fresh with EKF poses from after initialization
                    if req.data {
                        // Log init-to-tracking time
                        #[cfg(feature = "debug-output")]
                        {
                            if let Some(start) =
                                init_request_time_for_trigger.lock().unwrap().take()
                            {
                                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                                log_info!(NODE_NAME, "Init-to-tracking time: {:.2}ms", elapsed_ms);
                                // Write to debug file
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
        // This service is called by pose_initializer with an initial pose guess

        let ndt_align_srv = {
            let ndt_manager = Arc::clone(&ndt_manager);
            let map_points = Arc::clone(&map_points);
            let latest_sensor_points = Arc::clone(&latest_sensor_points);
            let params = Arc::clone(&params);
            let monte_carlo_pub = debug_pubs.monte_carlo_marker_pub.clone();

            node.create_service::<autoware_internal_localization_msgs::srv::PoseWithCovarianceStamped, _>(
                "ndt_align_srv",
                move |req: super::state::PoseWithCovSrvRequest, _info: rclrs::ServiceInfo| {
                    // Record init request time for end-to-end tracking
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

        // Map update service (triggers map update based on current position)
        let map_update_srv = {
            let map_module = Arc::clone(&map_module);
            let map_points = Arc::clone(&map_points);
            let ndt_manager = Arc::clone(&ndt_manager);
            let pose_buffer = Arc::clone(&pose_buffer);

            node.create_service::<std_srvs::srv::Trigger, _>(
                "map_update_srv",
                move |_req: super::state::TriggerRequest, _info: rclrs::ServiceInfo| {
                    services::on_map_update(&map_module, &map_points, &ndt_manager, &pose_buffer)
                },
            )?
        };

        // Clear debug file at startup (only with debug-output feature)
        #[cfg(feature = "debug-output")]
        {
            crate::io::debug_writer::clear_debug_file();
            log_info!(
                NODE_NAME,
                "Debug output cleared: {}",
                crate::io::debug_writer::debug_file_path()
            );
        }

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
