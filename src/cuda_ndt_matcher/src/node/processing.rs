use geometry_msgs::msg::PoseWithCovarianceStamped;
#[cfg(any(feature = "debug-output", feature = "debug-markers"))]
use ndt_cuda::AlignmentDebug;
#[cfg(feature = "debug-output")]
use rclrs::log_info;
use rclrs::{log_error, log_warn};
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::Instant;

use super::state::NODE_NAME;
use crate::alignment::{AlignResult, NdtManager};
use crate::io::params::NdtParams;
#[cfg(feature = "debug-output")]
use crate::{io::debug_writer, transform::pose_utils};

/// Output from alignment execution, capturing all values needed by the
/// debug publishing section of the on_points callback.
pub(crate) struct AlignmentOutput {
    pub result: AlignResult,
    pub exe_time_ms: f32,
    pub transform_prob: f64,
    pub transform_prob_before: f64,
    pub nvtl_score: f64,
    pub nvtl_before: f64,
    pub is_converged: bool,
    pub skipping_publish_num: i32,
    #[cfg(any(feature = "debug-output", feature = "debug-markers"))]
    pub alignment_debug: Option<AlignmentDebug>,
}

/// Run NDT alignment with convergence gating.
///
/// Handles:
/// - "Before" score computation (for diagnostics comparison)
/// - NDT alignment with feature-gated debug variants
/// - Score computation (transform probability and NVTL)
/// - Convergence gating (iteration check, oscillation check, score check)
///
/// Returns `None` if alignment fails (caller should return early).
#[allow(unused_variables)] // timestamp_ns used only with debug-output feature
pub(crate) fn run_alignment(
    manager: &mut NdtManager,
    sensor_points: &[[f32; 3]],
    map: &[[f32; 3]],
    initial_pose: &PoseWithCovarianceStamped,
    timestamp_ns: u64,
    params: &NdtParams,
    skip_counter: &AtomicI32,
) -> Option<AlignmentOutput> {
    // Debug: log pose being passed to NDT alignment (only with debug-output feature)
    #[cfg(feature = "debug-output")]
    {
        let p = &initial_pose.pose.pose.position;
        let (roll, pitch, yaw) = pose_utils::euler_from_pose(&initial_pose.pose.pose);
        log_info!(
            NODE_NAME,
            "[NDT_IN] ts_ns={} pos=({:.3}, {:.3}, {:.3}) rpy=({:.3}, {:.3}, {:.3}) n_pts={}",
            timestamp_ns,
            p.x,
            p.y,
            p.z,
            roll.to_degrees(),
            pitch.to_degrees(),
            yaw.to_degrees(),
            sensor_points.len()
        );
    }

    // Compute "before" scores at initial pose (for diagnostics comparison)
    let transform_prob_before = manager
        .evaluate_transform_probability(sensor_points, &initial_pose.pose.pose)
        .unwrap_or(0.0);
    let nvtl_before = manager
        .evaluate_nvtl(sensor_points, map, &initial_pose.pose.pose, 0.55)
        .unwrap_or(0.0);

    // Start execution timer here to measure only NDT alignment (matches Autoware's scope)
    let align_start_time = Instant::now();

    // With debug-output feature: collect and write debug data
    #[cfg(feature = "debug-output")]
    let (result, alignment_debug) =
        match manager.align_with_debug(sensor_points, map, &initial_pose.pose.pose, timestamp_ns) {
            Ok((r, debug)) => {
                // Write debug JSON to file
                if let Ok(json) = debug.to_json() {
                    debug_writer::append_debug_line(&json);
                }
                (r, Some(debug))
            }
            Err(e) => {
                log_error!(NODE_NAME, "NDT alignment failed: {e}");
                return None;
            }
        };

    // Without debug-output feature: just run alignment
    #[cfg(all(not(feature = "debug-output"), feature = "debug-markers"))]
    let (result, alignment_debug): (_, Option<AlignmentDebug>) =
        match manager.align(sensor_points, map, &initial_pose.pose.pose) {
            Ok(r) => (r, None),
            Err(e) => {
                log_error!(NODE_NAME, "NDT alignment failed: {e}");
                return None;
            }
        };

    // Without either debug feature: just run alignment
    #[cfg(all(not(feature = "debug-output"), not(feature = "debug-markers")))]
    let result = match manager.align(sensor_points, map, &initial_pose.pose.pose) {
        Ok(r) => r,
        Err(e) => {
            log_error!(NODE_NAME, "NDT alignment failed: {e}");
            return None;
        }
    };

    // Calculate execution time immediately after alignment (matches Autoware's scope)
    let exe_time_ms = align_start_time.elapsed().as_secs_f32() * 1000.0;

    // Debug: log NDT result (only with debug-output feature)
    #[cfg(feature = "debug-output")]
    {
        let p = &result.pose.position;
        let (roll, pitch, yaw) = pose_utils::euler_from_pose(&result.pose);
        log_info!(
            NODE_NAME,
            "[NDT_OUT] ts_ns={} pos=({:.3}, {:.3}, {:.3}) rpy=({:.3}, {:.3}, {:.3}) iter={} conv={} osc={}",
            timestamp_ns,
            p.x,
            p.y,
            p.z,
            roll.to_degrees(),
            pitch.to_degrees(),
            yaw.to_degrees(),
            result.iterations,
            result.converged,
            result.oscillation_count
        );
    }

    // ---- Compute scores for filtering decision ----
    // Like Autoware, we compute NVTL and transform_probability before deciding to publish

    // Compute transform probability (fitness score converted to probability)
    let transform_prob = (-result.score / 10.0).exp();

    // Compute NVTL score
    let nvtl_score = manager
        .evaluate_nvtl(sensor_points, map, &result.pose, 0.55)
        .unwrap_or(0.0);

    // ---- Convergence gating (matching Autoware's behavior) ----
    // Autoware gates pose publishing on three conditions:
    // 1. is_ok_iteration_num: did NOT hit max iterations (result.converged)
    // 2. is_local_optimal_solution_oscillation: oscillation count <= 10
    // 3. is_ok_score: score above threshold

    // Check 1: Max iterations (result.converged is false when max iterations reached)
    let is_ok_iteration_num = result.converged;

    // Check 2: Oscillation count (Autoware uses threshold of 10)
    const OSCILLATION_THRESHOLD: usize = 10;
    let is_ok_oscillation = result.oscillation_count <= OSCILLATION_THRESHOLD;

    // Check 3: Score threshold
    // converged_param_type: 0 = transform_probability, 1 = NVTL
    let (score_for_check, threshold, score_name) = if params.score.converged_param_type == 0 {
        (
            transform_prob,
            params.score.converged_param_transform_probability,
            "transform_probability",
        )
    } else {
        (
            nvtl_score,
            params
                .score
                .converged_param_nearest_voxel_transformation_likelihood,
            "NVTL",
        )
    };
    let is_ok_score = score_for_check >= threshold;

    // Combined convergence check (all three must pass)
    let is_converged = is_ok_iteration_num && is_ok_oscillation && is_ok_score;

    // Track consecutive skips for diagnostics and log reasons
    let skipping_publish_num = if !is_converged {
        let skips = skip_counter.fetch_add(1, Ordering::SeqCst) + 1;

        // Log specific reason(s) for skipping
        if !is_ok_iteration_num {
            log_warn!(
                NODE_NAME,
                "Max iterations reached: iter={}, skip_count={skips}",
                result.iterations
            );
        }
        if !is_ok_oscillation {
            log_warn!(
                NODE_NAME,
                "Oscillation detected: count={} > {OSCILLATION_THRESHOLD}, skip_count={skips}",
                result.oscillation_count
            );
        }
        if !is_ok_score {
            log_warn!(
                NODE_NAME,
                "Score below threshold: {score_name}={score_for_check:.3} < {threshold:.3}, skip_count={skips}"
            );
        }
        skips
    } else {
        skip_counter.store(0, Ordering::SeqCst);
        0
    };

    Some(AlignmentOutput {
        result,
        exe_time_ms,
        transform_prob,
        transform_prob_before,
        nvtl_score,
        nvtl_before,
        is_converged,
        skipping_publish_num,
        #[cfg(any(feature = "debug-output", feature = "debug-markers"))]
        alignment_debug,
    })
}
