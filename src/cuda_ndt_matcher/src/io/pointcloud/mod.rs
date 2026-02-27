//! Point cloud conversion and filtering.
//!
//! - [`cpu`]: PointCloud2 message parsing and construction (pure CPU)
//! - [`gpu`]: Sensor point filtering with GPU acceleration and CPU fallback

pub(crate) mod cpu;
pub(crate) mod gpu;

// Re-export commonly used items so callers can use `pointcloud::from_pointcloud2()` etc.
pub(crate) use cpu::from_pointcloud2;
pub(crate) use cpu::to_pointcloud2;
#[cfg(feature = "debug-markers")]
pub(crate) use cpu::to_pointcloud2_with_rgb;
pub(crate) use gpu::{filter_sensor_points, PointFilterParams};
