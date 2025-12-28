//! Segment detection for voxel boundary identification.
//!
//! After Morton-sorting points, segments are contiguous runs of points
//! with the same Morton code (i.e., same voxel). This module detects
//! segment boundaries to identify voxel extents.
//!
//! # Algorithm
//!
//! 1. **Mark boundaries**: For each element, check if it differs from previous
//! 2. **Prefix sum**: Compute segment IDs from boundary marks
//! 3. **Extract**: Gather (segment_id, start_index, end_index, morton_code)
//!
//! # Current Status
//!
//! GPU kernels are defined but require CubeCL type system fixes.
//! CPU reference implementations are provided and tested.

/// Result of segment detection.
#[derive(Debug)]
pub struct SegmentResult {
    /// Segment ID for each point (which voxel it belongs to).
    pub segment_ids: Vec<u32>,
    /// Start index of each segment (one per unique voxel).
    pub segment_starts: Vec<u32>,
    /// Morton code for each segment.
    pub segment_codes: Vec<u64>,
    /// Number of points.
    pub num_points: u32,
    /// Number of unique segments (voxels).
    pub num_segments: u32,
}

/// Detect segments in sorted Morton codes (CPU reference implementation).
///
/// # Arguments
/// * `sorted_codes` - Sorted Morton codes
///
/// # Returns
/// Segment information including starts, codes, and per-point segment IDs.
pub fn detect_segments_cpu(sorted_codes: &[u64]) -> SegmentResult {
    let n = sorted_codes.len();
    if n == 0 {
        return SegmentResult {
            segment_ids: Vec::new(),
            segment_starts: Vec::new(),
            segment_codes: Vec::new(),
            num_points: 0,
            num_segments: 0,
        };
    }

    // Compute segment IDs by running counter
    // Each point gets the ID of its segment (0-indexed)
    let mut segment_ids = vec![0u32; n];
    let mut current_segment = 0u32;
    let mut segment_starts = Vec::new();
    let mut segment_codes = Vec::new();

    // First point always starts segment 0
    segment_starts.push(0u32);
    segment_codes.push(sorted_codes[0]);

    for i in 1..n {
        if sorted_codes[i] != sorted_codes[i - 1] {
            // New segment
            current_segment += 1;
            segment_starts.push(i as u32);
            segment_codes.push(sorted_codes[i]);
        }
        segment_ids[i] = current_segment;
    }

    let num_segments = current_segment + 1;

    SegmentResult {
        segment_ids,
        segment_starts,
        segment_codes,
        num_points: n as u32,
        num_segments,
    }
}

/// Extended segment result with lengths.
#[derive(Debug)]
pub struct SegmentResultWithLengths {
    /// Base segment result.
    pub base: SegmentResult,
    /// Length of each segment (points per voxel).
    pub segment_lengths: Vec<u32>,
}

/// Detect segments and compute lengths (CPU reference implementation).
pub fn detect_segments_with_lengths_cpu(sorted_codes: &[u64]) -> SegmentResultWithLengths {
    let base = detect_segments_cpu(sorted_codes);

    if base.num_segments == 0 {
        return SegmentResultWithLengths {
            base,
            segment_lengths: Vec::new(),
        };
    }

    // Compute lengths
    let num_segments = base.num_segments as usize;
    let mut segment_lengths = vec![0u32; num_segments];
    for (i, len) in segment_lengths.iter_mut().enumerate() {
        let start = base.segment_starts[i];
        let end = if i + 1 < num_segments {
            base.segment_starts[i + 1]
        } else {
            base.num_points
        };
        *len = end - start;
    }

    SegmentResultWithLengths {
        base,
        segment_lengths,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_detection_cpu() {
        // Test case: 3 segments
        // sorted_codes: [1, 1, 1, 5, 5, 9, 9, 9, 9]
        // segment_ids: Each point's voxel ID
        //   Points 0,1,2 (code=1) -> segment 0
        //   Points 3,4 (code=5) -> segment 1
        //   Points 5,6,7,8 (code=9) -> segment 2
        let sorted_codes = vec![1u64, 1, 1, 5, 5, 9, 9, 9, 9];
        let result = detect_segments_cpu(&sorted_codes);

        // Each point's segment ID (which voxel it belongs to)
        assert_eq!(result.segment_ids, vec![0, 0, 0, 1, 1, 2, 2, 2, 2]);
        assert_eq!(result.segment_starts, vec![0, 3, 5]);
        assert_eq!(result.segment_codes, vec![1, 5, 9]);
        assert_eq!(result.num_segments, 3);
    }

    #[test]
    fn test_all_same_code() {
        let sorted_codes = vec![42u64; 100];
        let result = detect_segments_cpu(&sorted_codes);
        assert_eq!(result.num_segments, 1);
    }

    #[test]
    fn test_all_different_codes() {
        let sorted_codes: Vec<u64> = (0..50).collect();
        let result = detect_segments_cpu(&sorted_codes);
        assert_eq!(result.num_segments, 50);
    }

    #[test]
    fn test_segment_lengths() {
        let sorted_codes = vec![1u64, 1, 1, 5, 5, 9, 9, 9, 9];
        let result = detect_segments_with_lengths_cpu(&sorted_codes);

        assert_eq!(result.segment_lengths, vec![3, 2, 4]);
    }

    #[test]
    fn test_empty() {
        let result = detect_segments_cpu(&[]);
        assert_eq!(result.num_segments, 0);
        assert!(result.segment_ids.is_empty());
    }
}
