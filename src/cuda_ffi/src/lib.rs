//! FFI bindings for CUDA libraries used by cuda_ndt_matcher.
//!
//! This crate provides Rust bindings for:
//! - CUB DeviceRadixSort (GPU radix sort)
//! - CUB DeviceScan (prefix sum)
//! - CUB DeviceSelect (stream compaction)
//! - GPU segment detection for voxel boundaries
//!
//! # Example
//!
//! ```ignore
//! use cuda_ffi::{RadixSorter, SegmentDetector};
//!
//! let sorter = RadixSorter::new()?;
//! let (sorted_keys, sorted_values) = sorter.sort_pairs(&keys, &values)?;
//!
//! let detector = SegmentDetector::new()?;
//! let segments = detector.detect_segments(&sorted_keys)?;
//! ```

pub mod radix_sort;
pub mod segment_detect;
pub mod segmented_reduce;

pub use radix_sort::{
    radix_sort_temp_size, sort_pairs_inplace, CudaError, DeviceBuffer, RadixSorter,
};
pub use segment_detect::{
    detect_segments_inplace, segment_detect_temp_sizes, SegmentCounts, SegmentDetector,
    SegmentResult,
};
pub use segmented_reduce::{
    segmented_reduce_sum_f32_inplace, segmented_reduce_sum_f32_temp_size,
    segmented_reduce_sum_f64_inplace, segmented_reduce_sum_f64_temp_size, SegmentedReducer,
};
