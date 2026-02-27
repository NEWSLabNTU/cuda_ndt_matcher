mod alignment;
mod initial_pose;
mod io;
mod map;
mod node;
mod scoring;
mod transform;
mod visualization;

use anyhow::Result;
use node::NdtScanMatcherNode;
use rclrs::{log_info, Context, CreateBasicExecutor, RclrsErrorFilter, SpinOptions};

const NODE_NAME: &str = "ndt_scan_matcher";

fn main() -> Result<()> {
    let mut executor = Context::default_from_env()?.create_basic_executor();
    let node = executor.create_node(NODE_NAME)?;

    let _ndt_node = NdtScanMatcherNode::new(&node)?;

    log_info!(NODE_NAME, "Waiting for map and initial pose...");

    // Spin until shutdown (Ctrl-C or ROS shutdown)
    // Using default spin options which properly handles service responses
    executor.spin(SpinOptions::default()).first_error()?;

    log_info!(NODE_NAME, "Shutdown complete");
    Ok(())
}
