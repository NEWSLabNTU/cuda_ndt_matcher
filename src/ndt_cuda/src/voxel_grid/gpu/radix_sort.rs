//! GPU radix sort for Morton codes.
//!
//! Implements LSB (least-significant-bit) radix sort for 64-bit keys.
//! Uses a 4-bit radix (16 buckets) for each pass, requiring 16 passes total.
//!
//! # Algorithm
//!
//! For each 4-bit digit (LSB to MSB):
//! 1. **Histogram**: Count occurrences of each digit value per block
//! 2. **Scan**: Prefix sum on histograms to compute global offsets
//! 3. **Scatter**: Move elements to their sorted positions
//!
//! This is a stable sort, preserving relative order of equal keys.
//!
//! # Current Status
//!
//! GPU kernels are defined but require CubeCL type system fixes.
//! CPU reference implementations are provided and tested.

/// Radix (number of possible values per digit).
/// 4-bit radix = 16 values per digit.
const RADIX_BITS: u32 = 4;
const RADIX: usize = 1 << RADIX_BITS; // 16

/// Number of passes for 64-bit keys.
const NUM_PASSES: u32 = 64 / RADIX_BITS; // 16

/// Result of radix sort operation.
#[derive(Debug)]
pub struct RadixSortResult {
    /// Sorted keys (raw bytes, interpret as u64).
    pub keys: Vec<u8>,
    /// Reordered values (raw bytes, interpret as u32).
    pub values: Vec<u8>,
    /// Number of elements.
    pub num_elements: u32,
}

/// Perform radix sort on Morton codes with associated indices (CPU reference).
///
/// # Arguments
/// * `keys` - Morton codes to sort
/// * `values` - Associated values (original indices)
///
/// # Returns
/// Sorted keys and reordered values.
pub fn radix_sort_by_key_cpu(keys: &[u64], values: &[u32]) -> RadixSortResult {
    let n = keys.len();
    if n == 0 {
        return RadixSortResult {
            keys: Vec::new(),
            values: Vec::new(),
            num_elements: 0,
        };
    }

    let mut keys_a = keys.to_vec();
    let mut values_a = values.to_vec();
    let mut keys_b = vec![0u64; n];
    let mut values_b = vec![0u32; n];

    for pass in 0..NUM_PASSES {
        let shift = pass * RADIX_BITS;

        // Count histogram
        let mut hist = [0usize; RADIX];
        for &k in &keys_a {
            let digit = ((k >> shift) & (RADIX as u64 - 1)) as usize;
            hist[digit] += 1;
        }

        // Prefix sum
        let mut sum = 0;
        let mut offsets = [0usize; RADIX];
        for i in 0..RADIX {
            offsets[i] = sum;
            sum += hist[i];
        }

        // Scatter
        let mut counts = [0usize; RADIX];
        for i in 0..n {
            let digit = ((keys_a[i] >> shift) & (RADIX as u64 - 1)) as usize;
            let dest = offsets[digit] + counts[digit];
            counts[digit] += 1;
            keys_b[dest] = keys_a[i];
            values_b[dest] = values_a[i];
        }

        std::mem::swap(&mut keys_a, &mut keys_b);
        std::mem::swap(&mut values_a, &mut values_b);
    }

    // Convert to bytes
    let mut key_bytes = Vec::with_capacity(n * 8);
    let mut value_bytes = Vec::with_capacity(n * 4);
    for i in 0..n {
        key_bytes.extend_from_slice(&keys_a[i].to_le_bytes());
        value_bytes.extend_from_slice(&values_a[i].to_le_bytes());
    }

    RadixSortResult {
        keys: key_bytes,
        values: value_bytes,
        num_elements: n as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radix_sort_cpu() {
        let keys = vec![5u64, 3, 8, 1, 9, 2, 7, 4, 6, 0];
        let values: Vec<u32> = (0..10).collect();

        let result = radix_sort_by_key_cpu(&keys, &values);

        // Parse result keys
        let sorted_keys: Vec<u64> = result
            .keys
            .chunks(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect();

        assert_eq!(sorted_keys, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_radix_sort_preserves_order() {
        // Test stability: equal keys preserve original order
        let keys = vec![1u64, 1, 1, 1, 1];
        let values: Vec<u32> = vec![0, 1, 2, 3, 4];

        let result = radix_sort_by_key_cpu(&keys, &values);

        let sorted_values: Vec<u32> = result
            .values
            .chunks(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        // Stable sort should preserve order
        assert_eq!(sorted_values, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_radix_sort_empty() {
        let result = radix_sort_by_key_cpu(&[], &[]);
        assert_eq!(result.num_elements, 0);
    }
}
