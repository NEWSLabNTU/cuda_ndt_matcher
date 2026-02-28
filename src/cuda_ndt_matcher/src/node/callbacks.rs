use autoware_internal_debug_msgs::msg::{Float32Stamped, Int32Stamped};
use geometry_msgs::msg::{PoseArray, PoseStamped, PoseWithCovariance, PoseWithCovarianceStamped};
use rclrs::{log_debug, log_error, log_info, log_warn};
use sensor_msgs::msg::PointCloud2;
use std::{
    sync::{Arc, atomic::Ordering},
    time::Instant,
};
use std_msgs::msg::Header;
use visualization_msgs::msg::MarkerArray;

use super::{
    processing::{self, AlignmentOutput},
    publishers,
    state::{NODE_NAME, NdtScanMatcherNode, OnPointsContext},
};
use crate::{
    alignment::{batch::QueuedScan, covariance},
    io::{
        diagnostics::{DiagnosticLevel, ScanMatchingDiagnostics},
        pointcloud,
    },
    transform::{SmartPoseBuffer, pose_utils},
};

impl NdtScanMatcherNode {
    pub(crate) fn on_points(msg: PointCloud2, ctx: &OnPointsContext) {
        let _cb_num = ctx.callback_count.fetch_add(1, Ordering::SeqCst) + 1;
        let timestamp_ns = pose_utils::stamp_to_ns_u64(&msg.header.stamp);
        let sensor_time_ns = pose_utils::stamp_to_ns(&msg.header.stamp);

        // Stage 1: Convert and filter sensor points
        let sensor_points = match convert_and_filter_points(&msg) {
            Some(pts) => pts,
            None => return,
        };

        // Stage 2: Transform from sensor frame to base_link
        // Split into lookup + apply so TF lookup can potentially overlap with
        // point filtering in future pipeline optimizations (Phase 27.6a).
        let base_tf = lookup_base_transform(
            &msg.header.frame_id,
            &ctx.params.frame.base_frame,
            sensor_time_ns,
            &ctx.tf_handler,
        );
        let sensor_points = apply_base_transform(sensor_points, base_tf.as_ref());

        // Store sensor points for ndt_align service (before any early returns)
        ctx.latest_sensor_points
            .store(Arc::new(Some(sensor_points.clone())));

        if !ctx.enabled.load(Ordering::SeqCst) {
            return;
        }

        // Stage 3: Interpolate initial pose to sensor timestamp
        let interpolate_result = match interpolate_initial_pose(&ctx.pose_buffer, sensor_time_ns) {
            Some(r) => r,
            None => return,
        };
        let initial_pose = &interpolate_result.interpolated_pose;
        ctx.pose_buffer.pop_old(sensor_time_ns);

        // Stage 4: Update map if needed
        let current_position = pose_utils::position_from_pose_cov(initial_pose);
        update_map_if_needed(ctx, &current_position, &msg.header.stamp);

        // Get map points
        let map = ctx.map_points.load();
        let map = match map.as_ref() {
            Some(m) => m,
            None => {
                log_warn!(NODE_NAME, "No map loaded, skipping alignment");
                return;
            }
        };

        // Check minimum sensor point distance
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

        // Stage 5: Batch mode — enqueue and return
        if let Some(queue) = &ctx.scan_queue {
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
            return;
        }

        // Stage 6: Synchronous NDT alignment
        let mut manager = ctx.ndt_manager.lock();
        let output = match processing::run_alignment(
            &mut manager,
            &sensor_points,
            map,
            initial_pose,
            timestamp_ns,
            &ctx.params,
            &ctx.skip_counter,
        ) {
            Some(output) => output,
            None => return,
        };

        let header = Header {
            stamp: msg.header.stamp.clone(),
            frame_id: ctx.params.frame.map_frame.clone(),
        };

        // Stage 7: Publish converged pose, TF, covariance
        if output.is_converged {
            publish_converged_pose(ctx, &header, &output, &mut manager, &sensor_points, map);
        }

        // Stage 8: Debug publishers and diagnostics
        publish_debug_and_diagnostics(
            ctx,
            &msg,
            &header,
            &output,
            &interpolate_result,
            &sensor_points,
            map,
            &mut manager,
            max_dist,
        );
    }
}

/// Parse PointCloud2 message and apply sensor point filters.
fn convert_and_filter_points(msg: &PointCloud2) -> Option<Vec<[f32; 3]>> {
    let raw_points = match pointcloud::from_pointcloud2(msg) {
        Ok(pts) => pts,
        Err(e) => {
            log_error!(NODE_NAME, "Failed to convert point cloud: {e}");
            return None;
        }
    };

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

    Some(sensor_points)
}

/// Look up the sensor-to-base transform (TF lookup only, no point processing).
///
/// Split from the bulk transform to allow the TF lookup to potentially
/// overlap with point filtering in the pipeline (Phase 27.6a).
fn lookup_base_transform(
    sensor_frame: &str,
    base_frame: &str,
    stamp_ns: i64,
    tf_handler: &Arc<crate::transform::tf_handler::TfHandler>,
) -> Option<nalgebra::Isometry3<f64>> {
    if sensor_frame == base_frame {
        return Some(nalgebra::Isometry3::identity());
    }

    match tf_handler.lookup_transform(sensor_frame, base_frame, Some(stamp_ns)) {
        Some(tf) => {
            log_debug!(
                NODE_NAME,
                "TF lookup: {} -> {} OK",
                sensor_frame,
                base_frame
            );
            Some(tf)
        }
        None => {
            log_warn!(
                NODE_NAME,
                "TF not available: {} -> {}, using raw sensor frame",
                sensor_frame,
                base_frame
            );
            None
        }
    }
}

/// Apply the sensor-to-base transform to a point cloud.
///
/// If no transform is available (lookup returned None), returns points unchanged.
fn apply_base_transform(
    sensor_points: Vec<[f32; 3]>,
    tf: Option<&nalgebra::Isometry3<f64>>,
) -> Vec<[f32; 3]> {
    match tf {
        Some(tf) if *tf != nalgebra::Isometry3::identity() => {
            pose_utils::transform_points_f32(&sensor_points, tf)
        }
        _ => sensor_points,
    }
}

/// Interpolate initial pose from the pose buffer at the given sensor timestamp.
fn interpolate_initial_pose(
    pose_buffer: &Arc<SmartPoseBuffer>,
    sensor_time_ns: i64,
) -> Option<crate::transform::pose_buffer::InterpolateResult> {
    let result = pose_buffer.interpolate(sensor_time_ns);
    match result {
        Some(r) => {
            #[cfg(feature = "debug-output")]
            {
                let p = &r.interpolated_pose.pose.pose.position;
                let ts = &r.interpolated_pose.header.stamp;
                let (roll, pitch, yaw) =
                    pose_utils::euler_from_pose(&r.interpolated_pose.pose.pose);
                log_info!(
                    NODE_NAME,
                    "[INTERP] ts={}.{:09} pos=({:.3}, {:.3}, {:.3}) rpy=({:.3}, {:.3}, {:.3}) sensor_ts={}",
                    ts.sec,
                    ts.nanosec,
                    p.x,
                    p.y,
                    p.z,
                    roll.to_degrees(),
                    pitch.to_degrees(),
                    yaw.to_degrees(),
                    sensor_time_ns
                );
            }
            Some(r)
        }
        None => {
            if pose_buffer.len() < 2 {
                log_debug!(
                    NODE_NAME,
                    "Waiting for pose buffer to fill (size={}, need 2)",
                    pose_buffer.len()
                );
            } else {
                log_warn!(
                    NODE_NAME,
                    "Pose interpolation failed (validation error or timestamp mismatch)"
                );
            }
            None
        }
    }
}

/// Check if map needs updating and apply any pending updates.
fn update_map_if_needed(
    ctx: &OnPointsContext,
    position: &geometry_msgs::msg::Point,
    stamp: &builtin_interfaces::msg::Time,
) {
    // Request new map tiles if needed
    if ctx.map_module.should_update(position)
        && let Err(e) = ctx
            .map_loader
            .request_map_update(position, ctx.params.dynamic_map.map_radius as f32)
    {
        log_error!(NODE_NAME, "Failed to request map update: {e}");
    }

    // Apply any pending local map updates
    if let Some(filtered_map) = ctx.map_module.check_and_update(position) {
        ctx.map_points.store(Arc::new(Some(filtered_map.clone())));

        let debug_map_msg = pointcloud::to_pointcloud2(
            &filtered_map,
            &Header {
                stamp: stamp.clone(),
                frame_id: ctx.params.frame.map_frame.clone(),
            },
        );
        let _ = ctx
            .debug_pubs
            .debug_loaded_pointcloud_map_pub
            .publish(&debug_map_msg);

        let started = ctx
            .ndt_manager
            .start_background_update(filtered_map.clone());
        log_debug!(
            NODE_NAME,
            "Background NDT update started={started} with {} points",
            filtered_map.len()
        );
    }
}

/// Publish converged pose: covariance estimation, PoseStamped, TF, MULTI_NDT.
fn publish_converged_pose(
    ctx: &OnPointsContext,
    header: &Header,
    output: &AlignmentOutput,
    manager: &mut crate::alignment::manager::NdtManager,
    sensor_points: &[[f32; 3]],
    map: &[[f32; 3]],
) {
    let covariance_result = covariance::estimate_covariance_full(
        &ctx.params.covariance,
        &output.result.hessian,
        &output.result.pose,
        Some(manager),
        Some(sensor_points),
        Some(map),
    );

    let pose_msg = PoseStamped {
        header: header.clone(),
        pose: output.result.pose.clone(),
    };
    if let Err(e) = ctx.pose_pub.publish(&pose_msg) {
        log_error!(NODE_NAME, "Failed to publish pose: {e}");
    }

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

    publishers::publish_tf(
        &ctx.debug_pubs.tf_pub,
        &header.stamp,
        &output.result.pose,
        &ctx.params.frame.map_frame,
        &ctx.params.frame.ndt_base_frame,
    );

    if let Some(poses) = covariance_result.multi_ndt_poses {
        let pose_array_msg = PoseArray {
            header: header.clone(),
            poses,
        };
        let _ = ctx.debug_pubs.multi_ndt_pose_pub.publish(&pose_array_msg);
    }
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

/// Publish all debug metrics, aligned points, diagnostics.
#[allow(clippy::too_many_arguments)]
fn publish_debug_and_diagnostics(
    ctx: &OnPointsContext,
    msg: &PointCloud2,
    header: &Header,
    output: &AlignmentOutput,
    interpolate_result: &crate::transform::pose_buffer::InterpolateResult,
    sensor_points: &[[f32; 3]],
    map: &[[f32; 3]],
    manager: &mut crate::alignment::manager::NdtManager,
    max_dist: f32,
) {
    let initial_pose = &interpolate_result.interpolated_pose;

    // Track alignment count
    let align_num = ctx.align_count.fetch_add(1, Ordering::SeqCst) + 1;
    if align_num % 50 == 0 {
        let total_cb = ctx.callback_count.load(Ordering::SeqCst);
        log_info!(
            NODE_NAME,
            "Callback stats: total={total_cb}, aligned={align_num}"
        );
    }

    // Publish scalar debug metrics
    let _ = ctx.debug_pubs.exe_time_pub.publish(&Float32Stamped {
        stamp: msg.header.stamp.clone(),
        data: output.exe_time_ms,
    });
    let _ = ctx.debug_pubs.iteration_num_pub.publish(&Int32Stamped {
        stamp: msg.header.stamp.clone(),
        data: output.result.iterations,
    });
    let _ = ctx.debug_pubs.oscillation_count_pub.publish(&Int32Stamped {
        stamp: msg.header.stamp.clone(),
        data: output.result.oscillation_count as i32,
    });
    let _ = ctx
        .debug_pubs
        .transform_probability_pub
        .publish(&Float32Stamped {
            stamp: msg.header.stamp.clone(),
            data: output.transform_prob as f32,
        });
    let _ = ctx.debug_pubs.nvtl_pub.publish(&Float32Stamped {
        stamp: msg.header.stamp.clone(),
        data: output.nvtl_score as f32,
    });
    let _ = ctx.debug_pubs.initial_pose_cov_pub.publish(initial_pose);

    // Initial-to-result distance
    let distance = pose_utils::point_distance(
        &output.result.pose.position,
        &initial_pose.pose.pose.position,
    );
    let _ = ctx
        .debug_pubs
        .initial_to_result_distance_pub
        .publish(&Float32Stamped {
            stamp: msg.header.stamp.clone(),
            data: distance as f32,
        });

    // Distances from old/new interpolation poses to result
    {
        let distance_old = pose_utils::point_distance(
            &output.result.pose.position,
            &interpolate_result.old_pose.pose.pose.position,
        );
        let _ = ctx
            .debug_pubs
            .initial_to_result_distance_old_pub
            .publish(&Float32Stamped {
                stamp: msg.header.stamp.clone(),
                data: distance_old as f32,
            });

        let distance_new = pose_utils::point_distance(
            &output.result.pose.position,
            &interpolate_result.new_pose.pose.pose.position,
        );
        let _ = ctx
            .debug_pubs
            .initial_to_result_distance_new_pub
            .publish(&Float32Stamped {
                stamp: msg.header.stamp.clone(),
                data: distance_new as f32,
            });
    }

    // Relative pose (result relative to initial)
    let initial_isometry = pose_utils::isometry_from_pose(&initial_pose.pose.pose);
    let result_isometry = pose_utils::isometry_from_pose(&output.result.pose);
    let relative_isometry = result_isometry * initial_isometry.inverse();
    let relative_pose = pose_utils::pose_from_isometry(&relative_isometry);
    let _ = ctx
        .debug_pubs
        .initial_to_result_relative_pose_pub
        .publish(&PoseStamped {
            header: header.clone(),
            pose: relative_pose,
        });

    // NDT marker (pose history visualization)
    #[cfg(feature = "debug-markers")]
    let marker_array = if let Some(ref debug) = output.alignment_debug {
        publishers::create_pose_history_markers(header, debug)
    } else {
        let ndt_marker = publishers::create_pose_marker(header, &output.result.pose, 0);
        MarkerArray {
            markers: vec![ndt_marker],
        }
    };
    #[cfg(not(feature = "debug-markers"))]
    let marker_array = {
        let ndt_marker = publishers::create_pose_marker(header, &output.result.pose, 0);
        MarkerArray {
            markers: vec![ndt_marker],
        }
    };
    let _ = ctx.debug_pubs.ndt_marker_pub.publish(&marker_array);

    // Aligned points (sensor points transformed by result pose)
    let result_isometry = pose_utils::isometry_from_pose(&output.result.pose);
    let aligned_points = pose_utils::transform_points_f32(sensor_points, &result_isometry);
    let aligned_msg = pointcloud::to_pointcloud2(&aligned_points, header);
    let _ = ctx.debug_pubs.points_aligned_pub.publish(&aligned_msg);

    // Per-point score visualization (debug-markers feature)
    #[cfg(feature = "debug-markers")]
    if let Ok((score_points, scores)) =
        manager.compute_per_point_scores_for_visualization(sensor_points, &output.result.pose)
    {
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
            pointcloud::to_pointcloud2_with_rgb(&score_points, &rgb_values, header);
        let _ = ctx
            .debug_pubs
            .voxel_score_points_pub
            .publish(&score_cloud_msg);
    }

    // No-ground scoring (optional)
    if ctx.params.score.no_ground_points.enable {
        publish_no_ground_scores(ctx, msg, header, output, sensor_points, map, manager);
    }

    // Diagnostics
    publish_diagnostics(ctx, msg, output, sensor_points, max_dist, distance);
}

/// Compute and publish no-ground scores (filters ground points and rescores).
fn publish_no_ground_scores(
    ctx: &OnPointsContext,
    msg: &PointCloud2,
    header: &Header,
    output: &AlignmentOutput,
    sensor_points: &[[f32; 3]],
    map: &[[f32; 3]],
    manager: &mut crate::alignment::manager::NdtManager,
) {
    let pose_isometry = pose_utils::isometry_from_pose(&output.result.pose);
    let base_link_z = output.result.pose.position.z;
    let z_threshold = ctx
        .params
        .score
        .no_ground_points
        .z_margin_for_ground_removal as f64;

    // Transform once, produce both sensor-frame and map-frame filtered points.
    // This avoids the previous double-transform where filter transformed each point
    // to check Z, then transform_points_f32 transformed survivors again for publishing.
    let (no_ground_points, no_ground_aligned): (Vec<[f32; 3]>, Vec<[f32; 3]>) = sensor_points
        .iter()
        .filter_map(|pt| {
            let sensor_pt = nalgebra::Point3::new(pt[0] as f64, pt[1] as f64, pt[2] as f64);
            let map_pt = pose_isometry * sensor_pt;
            if map_pt.z - base_link_z > z_threshold {
                Some((*pt, [map_pt.x as f32, map_pt.y as f32, map_pt.z as f32]))
            } else {
                None
            }
        })
        .unzip();

    if no_ground_points.is_empty() {
        return;
    }

    let no_ground_tp = manager
        .evaluate_transform_probability(&no_ground_points, &output.result.pose)
        .unwrap_or(0.0);
    let no_ground_nvtl = manager
        .evaluate_nvtl(&no_ground_points, map, &output.result.pose, 0.55)
        .unwrap_or(0.0);
    let no_ground_cloud_msg = pointcloud::to_pointcloud2(&no_ground_aligned, header);
    let _ = ctx
        .debug_pubs
        .no_ground_points_aligned_pub
        .publish(&no_ground_cloud_msg);

    let _ = ctx
        .debug_pubs
        .no_ground_transform_probability_pub
        .publish(&Float32Stamped {
            stamp: msg.header.stamp.clone(),
            data: no_ground_tp as f32,
        });
    let _ = ctx.debug_pubs.no_ground_nvtl_pub.publish(&Float32Stamped {
        stamp: msg.header.stamp.clone(),
        data: no_ground_nvtl as f32,
    });
}

/// Collect and publish scan matching diagnostics.
fn publish_diagnostics(
    ctx: &OnPointsContext,
    msg: &PointCloud2,
    output: &AlignmentOutput,
    sensor_points: &[[f32; 3]],
    max_dist: f32,
    distance: f64,
) {
    let topic_time_stamp = msg.header.stamp.sec as f64 + msg.header.stamp.nanosec as f64 * 1e-9;

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
        sensor_points_delay_time_sec: 0.0,
        is_succeed_transform_sensor_points: true,
        sensor_points_max_distance: max_dist as f64,
        is_activated: true,
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

        diag.publish(msg.header.stamp.clone());
    }
}
