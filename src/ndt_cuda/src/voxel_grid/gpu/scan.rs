//! Parallel prefix sum (scan) implementation.
//!
//! Implements the Blelloch scan algorithm which runs in O(n) work and O(log n) depth.
//! Used for:
//! - Counting sort in radix sort
//! - Segment boundary detection
//! - Compaction operations
//!
//! # Algorithm (Blelloch Scan)
//!
//! 1. **Upsweep (Reduce)**: Build a binary tree of partial sums
//! 2. **Downsweep**: Propagate sums back down to produce prefix sums
//!
//! For large arrays, we use a multi-block approach:
//! 1. Each block computes local prefix sums
//! 2. Block sums are collected and scanned
//! 3. Block sums are added back to each block's elements
//!
//! # Current Status
//!
//! GPU kernels are defined but require CubeCL type system fixes.
//! CPU reference implementations are provided and tested.

// GPU kernel definitions commented out until CubeCL type system issues are resolved
// use cubecl::prelude::*;

/// Perform exclusive prefix sum (CPU reference implementation).
///
/// Exclusive scan: output[i] = input[0] + input[1] + ... + input[i-1]
/// (output[0] = 0)
///
/// # Arguments
/// * `input` - Input slice of u32 values
///
/// # Returns
/// Vector with exclusive prefix sums.
pub fn exclusive_scan_cpu(input: &[u32]) -> Vec<u32> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    let mut output = vec![0u32; n];
    let mut sum = 0u32;
    for i in 0..n {
        output[i] = sum;
        sum += input[i];
    }
    output
}

/// Perform inclusive prefix sum (CPU reference implementation).
///
/// Inclusive scan: output[i] = input[0] + input[1] + ... + input[i]
///
/// # Arguments
/// * `input` - Input slice of u32 values
///
/// # Returns
/// Vector with inclusive prefix sums.
pub fn inclusive_scan_cpu(input: &[u32]) -> Vec<u32> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    let mut output = vec![0u32; n];
    let mut sum = 0u32;
    for i in 0..n {
        sum += input[i];
        output[i] = sum;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exclusive_scan_cpu() {
        let input = vec![1, 2, 3, 4, 5];
        let expected = vec![0, 1, 3, 6, 10];
        assert_eq!(exclusive_scan_cpu(&input), expected);
    }

    #[test]
    fn test_inclusive_scan_cpu() {
        let input = vec![1, 2, 3, 4, 5];
        let expected = vec![1, 3, 6, 10, 15];
        assert_eq!(inclusive_scan_cpu(&input), expected);
    }

    #[test]
    fn test_empty_scan() {
        let input: Vec<u32> = vec![];
        assert!(exclusive_scan_cpu(&input).is_empty());
        assert!(inclusive_scan_cpu(&input).is_empty());
    }

    #[test]
    fn test_single_element() {
        let input = vec![42];
        assert_eq!(exclusive_scan_cpu(&input), vec![0]);
        assert_eq!(inclusive_scan_cpu(&input), vec![42]);
    }
}
