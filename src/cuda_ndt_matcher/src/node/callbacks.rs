use autoware_internal_debug_msgs::msg::{Float32Stamped, Int32Stamped};
use geometry_msgs::msg::{PoseArray, PoseStamped, PoseWithCovariance, PoseWithCovarianceStamped};
use nalgebra::Vector3;
use rclrs::{log_debug, log_error, log_info, log_warn};
use sensor_msgs::msg::PointCloud2;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use std_msgs::msg::Header;
use visualization_msgs::msg::MarkerArray;

use super::state::{NdtScanMatcherNode, OnPointsContext, NODE_NAME};
use super::{processing, publishers};
use crate::alignment::batch::QueuedScan;
use crate::alignment::covariance;
use crate::io::diagnostics::{DiagnosticLevel, ScanMatchingDiagnostics};
use crate::io::pointcloud;
use crate::transform::pose_utils;

impl NdtScanMatcherNode {
    pub(crate) fn on_points(msg: PointCloud2, ctx: &OnPointsContext) {
        // Track callback invocation
        let _cb_num = ctx.callback_count.fetch_add(1, Ordering::SeqCst) + 1;

        // Extract timestamp for debug output
        let timestamp_ns =
            msg.header.stamp.sec as u64 * 1_000_000_000 + msg.header.stamp.nanosec as u64;

        // Note: No explicit deduplication needed - QoS KeepLast(1) ensures we only
        // process the latest message, matching Autoware's approach.

        // Convert sensor points first - needed for align service even before we have initial pose
        let raw_points = match pointcloud::from_pointcloud2(&msg) {
            Ok(pts) => pts,
            Err(e) => {
                log_error!(NODE_NAME, "Failed to convert point cloud: {e}");
                return;
            }
        };

        // Note: Sensor point filtering (distance, z-height, downsampling) is handled
        // upstream by pointcloud_preprocessor. We use default (no-op) filtering here.
        let filter_params = pointcloud::PointFilterParams::default();
        let filter_result = pointcloud::filter_sensor_points(&raw_points, &filter_params);
        let sensor_points = filter_result.points;

        if sensor_points.len() < raw_points.len() {
            let gpu_str = if filter_result.used_gpu { "GPU" } else { "CPU" };
            log_debug!(
                NODE_NAME,
                "Filtered sensor points: {} -> {} (dist:{}, z:{}, downsample:{}) [{}]",
                raw_points.len(),
                sensor_points.len(),
                filter_result.removed_by_distance,
                filter_result.removed_by_z,
                filter_result.removed_by_downsampling,
                gpu_str
            );
        }

        // Transform sensor points from sensor frame to base_link
        // The sensor frame comes from the PointCloud2 header, target is base_frame from params
        let sensor_frame = &msg.header.frame_id;
        let base_frame = &ctx.params.frame.base_frame;
        let stamp_ns =
            msg.header.stamp.sec as i64 * 1_000_000_000 + msg.header.stamp.nanosec as i64;

        let sensor_points = if sensor_frame != base_frame {
            match ctx.tf_handler.transform_points(
                &sensor_points,
                sensor_frame,
                base_frame,
                Some(stamp_ns),
            ) {
                Some(transformed) => {
                    log_debug!(
                        NODE_NAME,
                        "Transformed {} points: {} -> {}",
                        transformed.len(),
                        sensor_frame,
                        base_frame
                    );
                    transformed
                }
                None => {
                    // TF not available yet - use points as-is with warning
                    log_warn!(
                        NODE_NAME,
                        "TF not available: {} -> {}, using raw sensor frame",
                        sensor_frame,
                        base_frame
                    );
                    sensor_points
                }
            }
        } else {
            // Already in base_frame, no transform needed
            sensor_points
        };

        // Always store sensor points for initial pose estimation service (ndt_align_srv)
        // This must happen before any early returns so the align service can work
        ctx.latest_sensor_points
            .store(Arc::new(Some(sensor_points.clone())));

        // Check if enabled for regular NDT alignment
        if !ctx.enabled.load(Ordering::SeqCst) {
            return;
        }

        // Get initial pose via interpolation to match sensor timestamp
        // This implements Autoware's SmartPoseBuffer behavior for better timestamp alignment
        let sensor_time_ns =
            msg.header.stamp.sec as i64 * 1_000_000_000 + msg.header.stamp.nanosec as i64;

        let interpolate_result = ctx.pose_buffer.interpolate(sensor_time_ns);
        let initial_pose = match &interpolate_result {
            Some(result) => {
                // Debug: log interpolated pose (only with debug-output feature)
                #[cfg(feature = "debug-output")]
                {
                    let p = &result.interpolated_pose.pose.pose.position;
                    let ts = &result.interpolated_pose.header.stamp;
                    let (roll, pitch, yaw) =
                        pose_utils::euler_from_pose(&result.interpolated_pose.pose.pose);
                    log_info!(
                        NODE_NAME,
                        "[INTERP] ts={}.{:09} pos=({:.3}, {:.3}, {:.3}) rpy=({:.3}, {:.3}, {:.3}) sensor_ts={}",
                        ts.sec, ts.nanosec,
                        p.x, p.y, p.z,
                        roll.to_degrees(), pitch.to_degrees(), yaw.to_degrees(),
                        sensor_time_ns
                    );
                }
                &result.interpolated_pose
            }
            None => {
                // Interpolation failed - need at least 2 poses, or validation failed
                if ctx.pose_buffer.len() < 2 {
                    log_debug!(
                        NODE_NAME,
                        "Waiting for pose buffer to fill (size={}, need 2)",
                        ctx.pose_buffer.len()
                    );
                } else {
                    log_warn!(
                        NODE_NAME,
                        "Pose interpolation failed (validation error or timestamp mismatch)"
                    );
                }
                return;
            }
        };

        // Pop old poses to prevent unbounded buffer growth
        ctx.pose_buffer.pop_old(sensor_time_ns);

        // Note: Early alignments may have roll=0, pitch=0 (unrefined initial pose)
        // before EKF has fused any NDT output. These alignments may have indefinite
        // Hessians, but the regularization in newton.rs handles this correctly.
        // We process them anyway to bootstrap the EKF with NDT data.

        // Check if map needs updating based on current position
        // This implements Autoware's dynamic map loading behavior
        let current_position = geometry_msgs::msg::Point {
            x: initial_pose.pose.pose.position.x,
            y: initial_pose.pose.pose.position.y,
            z: initial_pose.pose.pose.position.z,
        };

        // Check if we should request new map tiles via service
        // This is non-blocking - the callback will update map_module when response arrives
        if ctx.map_module.should_update(&current_position) {
            if let Err(e) = ctx
                .map_loader
                .request_map_update(&current_position, ctx.params.dynamic_map.map_radius as f32)
            {
                log_error!(NODE_NAME, "Failed to request map update: {e}");
            }
        }

        // Check and apply any pending updates from the map module (local filtering)
        if let Some(filtered_map) = ctx.map_module.check_and_update(&current_position) {
            // Map was updated - refresh the shared map points
            ctx.map_points.store(Arc::new(Some(filtered_map.clone())));

            // Publish debug map for visualization
            let debug_map_msg = pointcloud::to_pointcloud2(
                &filtered_map,
                &Header {
                    stamp: msg.header.stamp.clone(),
                    frame_id: ctx.params.frame.map_frame.clone(),
                },
            );
            let _ = ctx
                .debug_pubs
                .debug_loaded_pointcloud_map_pub
                .publish(&debug_map_msg);

            // Start non-blocking NDT target update in background thread
            let started = ctx
                .ndt_manager
                .start_background_update(filtered_map.clone());
            log_debug!(
                NODE_NAME,
                "Background NDT update started={started} with {} points",
                filtered_map.len()
            );
        }

        // Get map points (may have been updated above)
        let map = ctx.map_points.load();
        let map = match map.as_ref() {
            Some(m) => m,
            None => {
                log_warn!(NODE_NAME, "No map loaded, skipping alignment");
                return;
            }
        };

        // Check minimum distance
        let max_dist = sensor_points
            .iter()
            .map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt())
            .fold(0.0f32, f32::max);

        if max_dist < ctx.params.sensor_points.required_distance {
            log_warn!(
                NODE_NAME,
                "Sensor points max distance {max_dist:.1}m < required {:.1}m",
                ctx.params.sensor_points.required_distance
            );
            return;
        }

        // ---- Batch Mode: Enqueue scan and return ----
        // If batch processing is enabled, enqueue the scan for parallel GPU processing
        // and return immediately. Results will be published asynchronously by the
        // scan queue's result callback.
        if let Some(queue) = &ctx.scan_queue {
            // Convert initial pose to Isometry3
            let initial_isometry = pose_utils::isometry_from_pose(&initial_pose.pose.pose);

            let queued_scan = QueuedScan {
                points: sensor_points.clone(),
                initial_pose: initial_isometry,
                timestamp: msg.header.stamp.clone(),
                timestamp_ns,
                header: msg.header.clone(),
                arrival_time: Instant::now(),
            };

            let enqueued = queue.enqueue(queued_scan);
            if enqueued {
                log_debug!(
                    NODE_NAME,
                    "Scan enqueued for batch processing: ts_ns={}, n_pts={}",
                    timestamp_ns,
                    sensor_points.len()
                );
            }

            // Return early - result will be published by the scan queue callback
            return;
        }

        // ---- Synchronous Mode: Run NDT alignment directly ----

        // Get lock on active NDT manager (also checks for pending swap from background update)
        let mut manager = ctx.ndt_manager.lock();

        // Run alignment and convergence gating
        let output = match processing::run_alignment(
            &mut *manager,
            &sensor_points,
            map,
            initial_pose,
            timestamp_ns,
            &ctx.params,
            &ctx.skip_counter,
        ) {
            Some(output) => output,
            None => return, // Alignment failed
        };

        // Create output header (needed for debug publishers even if we skip pose publishing)
        let header = Header {
            stamp: msg.header.stamp.clone(),
            frame_id: ctx.params.frame.map_frame.clone(),
        };

        // Only publish pose if all convergence conditions pass
        if output.is_converged {
            // Estimate covariance based on configured mode
            // For MULTI_NDT modes, we use parallel batch evaluation (Rayon)
            // NOTE: We reuse the manager lock from the alignment - don't try to lock again!
            let covariance_result = covariance::estimate_covariance_full(
                &ctx.params.covariance,
                &output.result.hessian,
                &output.result.pose,
                Some(&mut *manager), // Reuse existing lock to avoid deadlock
                Some(&sensor_points),
                Some(map),
            );

            // Publish PoseStamped
            let pose_msg = PoseStamped {
                header: header.clone(),
                pose: output.result.pose.clone(),
            };
            if let Err(e) = ctx.pose_pub.publish(&pose_msg) {
                log_error!(NODE_NAME, "Failed to publish pose: {e}");
            }

            // Publish PoseWithCovarianceStamped with estimated covariance
            let pose_cov_msg = PoseWithCovarianceStamped {
                header: header.clone(),
                pose: PoseWithCovariance {
                    pose: output.result.pose.clone(),
                    covariance: covariance_result.covariance,
                },
            };
            if let Err(e) = ctx.pose_cov_pub.publish(&pose_cov_msg) {
                log_error!(NODE_NAME, "Failed to publish pose with covariance: {e}");
            }

            // Publish TF transform (map -> ndt_base_link)
            // This matches Autoware's publish_tf() behavior
            publishers::publish_tf(
                &ctx.debug_pubs.tf_pub,
                &msg.header.stamp,
                &output.result.pose,
                &ctx.params.frame.map_frame,
                &ctx.params.frame.ndt_base_frame,
            );

            // Publish MULTI_NDT poses for debug visualization (only for MULTI_NDT modes)
            if let Some(poses) = covariance_result.multi_ndt_poses {
                let pose_array_msg = PoseArray {
                    header: header.clone(),
                    poses,
                };
                let _ = ctx.debug_pubs.multi_ndt_pose_pub.publish(&pose_array_msg);
            }

            // Publish MULTI_NDT initial poses for debug visualization
            if let Some(poses) = covariance_result.multi_initial_poses {
                let pose_array_msg = PoseArray {
                    header: header.clone(),
                    poses,
                };
                let _ = ctx
                    .debug_pubs
                    .multi_initial_pose_pub
                    .publish(&pose_array_msg);
            }
        }

        // ---- Debug Publishers (always publish for monitoring) ----

        // Track successful alignment
        let align_num = ctx.align_count.fetch_add(1, Ordering::SeqCst) + 1;

        // Log periodic summary every 50 alignments
        if align_num % 50 == 0 {
            let total_cb = ctx.callback_count.load(Ordering::SeqCst);
            log_info!(
                NODE_NAME,
                "Callback stats: total={total_cb}, aligned={align_num}"
            );
        }

        // Publish execution time
        let exe_time_msg = Float32Stamped {
            stamp: msg.header.stamp.clone(),
            data: output.exe_time_ms,
        };
        let _ = ctx.debug_pubs.exe_time_pub.publish(&exe_time_msg);

        // Publish iteration count
        let iteration_msg = Int32Stamped {
            stamp: msg.header.stamp.clone(),
            data: output.result.iterations,
        };
        let _ = ctx.debug_pubs.iteration_num_pub.publish(&iteration_msg);

        // Publish oscillation count (detects if optimizer is bouncing between poses)
        let oscillation_msg = Int32Stamped {
            stamp: msg.header.stamp.clone(),
            data: output.result.oscillation_count as i32,
        };
        let _ = ctx
            .debug_pubs
            .oscillation_count_pub
            .publish(&oscillation_msg);

        // Publish transform probability
        let transform_prob_msg = Float32Stamped {
            stamp: msg.header.stamp.clone(),
            data: output.transform_prob as f32,
        };
        let _ = ctx
            .debug_pubs
            .transform_probability_pub
            .publish(&transform_prob_msg);

        // Publish NVTL score
        let nvtl_msg = Float32Stamped {
            stamp: msg.header.stamp.clone(),
            data: output.nvtl_score as f32,
        };
        let _ = ctx.debug_pubs.nvtl_pub.publish(&nvtl_msg);

        // Publish initial pose with covariance
        let _ = ctx.debug_pubs.initial_pose_cov_pub.publish(initial_pose);

        // Calculate initial to result distance
        let dx = output.result.pose.position.x - initial_pose.pose.pose.position.x;
        let dy = output.result.pose.position.y - initial_pose.pose.pose.position.y;
        let dz = output.result.pose.position.z - initial_pose.pose.pose.position.z;
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
        let distance_msg = Float32Stamped {
            stamp: msg.header.stamp.clone(),
            data: distance as f32,
        };
        let _ = ctx
            .debug_pubs
            .initial_to_result_distance_pub
            .publish(&distance_msg);

        // Calculate distance from old/new interpolation poses to result
        // (interpolate_result is guaranteed to be Some here since we returned early otherwise)
        if let Some(ref interp) = interpolate_result {
            // Distance from old pose (older of the two bracketing poses) to result
            let dx_old = output.result.pose.position.x - interp.old_pose.pose.pose.position.x;
            let dy_old = output.result.pose.position.y - interp.old_pose.pose.pose.position.y;
            let dz_old = output.result.pose.position.z - interp.old_pose.pose.pose.position.z;
            let distance_old = (dx_old * dx_old + dy_old * dy_old + dz_old * dz_old).sqrt();
            let _ = ctx
                .debug_pubs
                .initial_to_result_distance_old_pub
                .publish(&Float32Stamped {
                    stamp: msg.header.stamp.clone(),
                    data: distance_old as f32,
                });

            // Distance from new pose (newer of the two bracketing poses) to result
            let dx_new = output.result.pose.position.x - interp.new_pose.pose.pose.position.x;
            let dy_new = output.result.pose.position.y - interp.new_pose.pose.pose.position.y;
            let dz_new = output.result.pose.position.z - interp.new_pose.pose.pose.position.z;
            let distance_new = (dx_new * dx_new + dy_new * dy_new + dz_new * dz_new).sqrt();
            let _ = ctx
                .debug_pubs
                .initial_to_result_distance_new_pub
                .publish(&Float32Stamped {
                    stamp: msg.header.stamp.clone(),
                    data: distance_new as f32,
                });
        }

        // Publish relative pose (result relative to initial)
        // Compute actual relative transform: relative = result * initial^(-1)
        // This gives the transform that takes you from initial pose to result pose
        let initial_isometry = pose_utils::isometry_from_pose(&initial_pose.pose.pose);
        let result_isometry_rel = pose_utils::isometry_from_pose(&output.result.pose);

        // Compute relative transform: result * initial^(-1)
        let relative_isometry = result_isometry_rel * initial_isometry.inverse();
        let relative_pose = pose_utils::pose_from_isometry(&relative_isometry);

        let relative_pose_msg = PoseStamped {
            header: header.clone(),
            pose: relative_pose,
        };
        let _ = ctx
            .debug_pubs
            .initial_to_result_relative_pose_pub
            .publish(&relative_pose_msg);

        // Publish NDT marker (pose history visualization)
        // When debug-markers is enabled, publish pose history from debug data.
        // Otherwise, just publish the final pose.
        #[cfg(feature = "debug-markers")]
        let marker_array = if let Some(ref debug) = output.alignment_debug {
            publishers::create_pose_history_markers(&header, debug)
        } else {
            let ndt_marker = publishers::create_pose_marker(&header, &output.result.pose, 0);
            MarkerArray {
                markers: vec![ndt_marker],
            }
        };

        #[cfg(not(feature = "debug-markers"))]
        let marker_array = {
            let ndt_marker = publishers::create_pose_marker(&header, &output.result.pose, 0);
            MarkerArray {
                markers: vec![ndt_marker],
            }
        };

        let _ = ctx.debug_pubs.ndt_marker_pub.publish(&marker_array);

        // Publish aligned points (transformed sensor points with proper rotation)
        // Build isometry from result pose for proper point transformation
        let result_isometry = pose_utils::isometry_from_pose(&output.result.pose);

        let aligned_points: Vec<[f32; 3]> = sensor_points
            .iter()
            .map(|p| {
                // Transform point by result pose (rotation + translation)
                let sensor_pt = Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64);
                let map_pt = result_isometry * nalgebra::Point3::from(sensor_pt);
                [map_pt.x as f32, map_pt.y as f32, map_pt.z as f32]
            })
            .collect();
        let aligned_msg = pointcloud::to_pointcloud2(&aligned_points, &header);
        let _ = ctx.debug_pubs.points_aligned_pub.publish(&aligned_msg);

        // ---- Per-Point Score Visualization (requires debug-markers feature) ----
        // Compute per-point NDT scores and publish as RGB-colored point cloud.
        // This matches Autoware's voxel_score_points output for debugging.
        #[cfg(feature = "debug-markers")]
        if let Ok((score_points, scores)) =
            manager.compute_per_point_scores_for_visualization(&sensor_points, &output.result.pose)
        {
            // Convert scores to RGB colors using Autoware's color scheme
            let rgb_values: Vec<u32> = scores
                .iter()
                .map(|&score| {
                    ndt_cuda::scoring::color_to_rgb_packed(&ndt_cuda::scoring::ndt_score_to_color(
                        score,
                        ndt_cuda::scoring::DEFAULT_SCORE_LOWER,
                        ndt_cuda::scoring::DEFAULT_SCORE_UPPER,
                    ))
                })
                .collect();

            let score_cloud_msg =
                pointcloud::to_pointcloud2_with_rgb(&score_points, &rgb_values, &header);
            let _ = ctx
                .debug_pubs
                .voxel_score_points_pub
                .publish(&score_cloud_msg);
        }

        // ---- No-Ground Scoring (optional) ----
        // When enabled, filters out ground points and computes scores on the remaining points.
        // Ground is defined as points with transformed_z - base_link_z <= z_margin.
        if ctx.params.score.no_ground_points.enable {
            // Build isometry from result pose for transforming points
            let pose_isometry = pose_utils::isometry_from_pose(&output.result.pose);
            let base_link_z = output.result.pose.position.z;
            let z_threshold = ctx
                .params
                .score
                .no_ground_points
                .z_margin_for_ground_removal as f64;

            // Filter sensor points: keep those whose transformed z is above ground threshold
            let no_ground_points: Vec<[f32; 3]> = sensor_points
                .iter()
                .filter(|pt| {
                    // Transform point to map frame
                    let sensor_pt = Vector3::new(pt[0] as f64, pt[1] as f64, pt[2] as f64);
                    let map_pt = pose_isometry * nalgebra::Point3::from(sensor_pt);
                    // Keep if point_z - base_link_z > threshold
                    map_pt.z - base_link_z > z_threshold
                })
                .copied()
                .collect();

            if !no_ground_points.is_empty() {
                // Compute scores on filtered (non-ground) points
                let no_ground_tp = manager
                    .evaluate_transform_probability(&no_ground_points, &output.result.pose)
                    .unwrap_or(0.0);
                let no_ground_nvtl = manager
                    .evaluate_nvtl(&no_ground_points, map, &output.result.pose, 0.55)
                    .unwrap_or(0.0);

                // Publish filtered point cloud (in map frame for visualization)
                let no_ground_aligned: Vec<[f32; 3]> = no_ground_points
                    .iter()
                    .map(|pt| {
                        let sensor_pt = Vector3::new(pt[0] as f64, pt[1] as f64, pt[2] as f64);
                        let map_pt = pose_isometry * nalgebra::Point3::from(sensor_pt);
                        [map_pt.x as f32, map_pt.y as f32, map_pt.z as f32]
                    })
                    .collect();
                let no_ground_cloud_msg = pointcloud::to_pointcloud2(&no_ground_aligned, &header);
                let _ = ctx
                    .debug_pubs
                    .no_ground_points_aligned_pub
                    .publish(&no_ground_cloud_msg);

                // Publish no-ground transform probability
                let no_ground_tp_msg = Float32Stamped {
                    stamp: msg.header.stamp.clone(),
                    data: no_ground_tp as f32,
                };
                let _ = ctx
                    .debug_pubs
                    .no_ground_transform_probability_pub
                    .publish(&no_ground_tp_msg);

                // Publish no-ground NVTL
                let no_ground_nvtl_msg = Float32Stamped {
                    stamp: msg.header.stamp.clone(),
                    data: no_ground_nvtl as f32,
                };
                let _ = ctx
                    .debug_pubs
                    .no_ground_nvtl_pub
                    .publish(&no_ground_nvtl_msg);
            }
        }

        // ---- Diagnostics ----
        // Collect and publish scan matching diagnostics
        let topic_time_stamp = msg.header.stamp.sec as f64 + msg.header.stamp.nanosec as f64 * 1e-9;

        // Extract per-iteration arrays from AlignmentDebug if available
        #[cfg(feature = "debug-iterations")]
        let (tp_array, nvtl_array) = output
            .alignment_debug
            .as_ref()
            .map(|d| {
                let tp = if d.transform_probability_array.is_empty() {
                    None
                } else {
                    Some(d.transform_probability_array.clone())
                };
                let nvtl = if d.nearest_voxel_transformation_likelihood_array.is_empty() {
                    None
                } else {
                    Some(d.nearest_voxel_transformation_likelihood_array.clone())
                };
                (tp, nvtl)
            })
            .unwrap_or((None, None));
        #[cfg(not(feature = "debug-iterations"))]
        let (tp_array, nvtl_array): (Option<Vec<f64>>, Option<Vec<f64>>) = (None, None);

        let scan_diag = ScanMatchingDiagnostics {
            topic_time_stamp,
            sensor_points_size: sensor_points.len(),
            sensor_points_delay_time_sec: 0.0, // Would need current time to compute
            is_succeed_transform_sensor_points: true,
            sensor_points_max_distance: max_dist as f64,
            is_activated: true, // We're here, so we're activated
            is_succeed_interpolate_initial_pose: true,
            is_set_map_points: true,
            iteration_num: output.result.iterations,
            oscillation_count: output.result.oscillation_count,
            transform_probability: output.transform_prob,
            nearest_voxel_transformation_likelihood: output.nvtl_score,
            transform_probability_before: output.transform_prob_before,
            nearest_voxel_transformation_likelihood_before: output.nvtl_before,
            distance_initial_to_result: distance,
            execution_time_ms: output.exe_time_ms as f64,
            skipping_publish_num: output.skipping_publish_num,
            transform_probability_array: tp_array,
            nearest_voxel_transformation_likelihood_array: nvtl_array,
        };

        {
            let mut diag = ctx.diagnostics.lock();
            scan_diag.apply_to(diag.scan_matching_mut());

            // Add map update diagnostics
            let map_status = ctx.map_loader.get_status();
            let map_diag = diag.map_update_mut();
            map_diag.clear();
            map_diag.add_key_value(
                "is_succeed_call_pcd_loader",
                map_status.last_request_success,
            );
            map_diag.add_key_value("pcd_loader_service_available", map_status.service_available);
            map_diag.add_key_value("tiles_loaded", ctx.map_module.tile_count());
            map_diag.add_key_value("tiles_added", map_status.tiles_added);
            map_diag.add_key_value("tiles_removed", map_status.tiles_removed);
            map_diag.add_key_value("points_added", map_status.points_added);
            if let Some(err) = &map_status.error_message {
                map_diag.add_key_value("error_message", err);
                map_diag.set_level_and_message(DiagnosticLevel::Warn, err);
            } else if !map_status.service_available && ctx.map_module.tile_count() == 0 {
                map_diag.set_level_and_message(
                    DiagnosticLevel::Warn,
                    "pcd_loader_service not available, no map loaded",
                );
            } else {
                map_diag.set_level_and_message(DiagnosticLevel::Ok, "OK");
            }

            diag.publish(msg.header.stamp);
        }
    }
}
