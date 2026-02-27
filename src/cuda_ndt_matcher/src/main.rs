mod covariance;
mod debug_writer;
mod diagnostics;
mod dual_ndt_manager;
mod initial_pose;
mod map_module;
mod ndt_manager;
mod node;
mod nvtl;
mod params;
mod particle;
mod pointcloud;
mod pose_buffer;
mod pose_utils;
mod scan_queue;
mod tf_handler;
mod tpe;
mod visualization;

use anyhow::Result;
use node::NdtScanMatcherNode;
use rclrs::{log_info, Context, CreateBasicExecutor, RclrsErrorFilter, SpinOptions};

const NODE_NAME: &str = "ndt_scan_matcher";

fn main() -> Result<()> {
    let mut executor = Context::default_from_env()?.create_basic_executor();
    let node = executor.create_node(NODE_NAME)?;

    let _ndt_node = NdtScanMatcherNode::new(&node)?;

    // TODO: For testing, load a dummy map
    // In production, this would come from pcd_loader_service
    log_info!(NODE_NAME, "Waiting for map and initial pose...");

    // Spin until shutdown (Ctrl-C or ROS shutdown)
    // Using default spin options which properly handles service responses
    executor.spin(SpinOptions::default()).first_error()?;

    log_info!(NODE_NAME, "Shutdown complete");
    Ok(())
}
