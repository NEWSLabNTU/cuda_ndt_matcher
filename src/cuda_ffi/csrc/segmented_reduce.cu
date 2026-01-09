// CUB DeviceSegmentedReduce wrapper for Rust FFI
//
// CUB is a header-only C++ template library. This file instantiates the
// templates for the specific types we need and provides a C interface.

#include <cub/cub.cuh>
#include <cuda_runtime.h>

extern "C" {

// Error codes matching cudaError_t
typedef int CudaError;

/// Query the temporary storage size needed for segmented reduce sum (f32).
///
/// # Arguments
/// * `temp_storage_bytes` - Output: required temporary storage size
/// * `num_items` - Total number of items across all segments
/// * `num_segments` - Number of segments to reduce
///
/// # Returns
/// cudaSuccess (0) on success, error code otherwise.
CudaError cub_segmented_reduce_sum_f32_temp_size(
    size_t* temp_storage_bytes,
    int num_items,
    int num_segments
) {
    // Use placeholder offsets for size query
    return cub::DeviceSegmentedReduce::Sum(
        nullptr,                    // d_temp_storage (nullptr to query size)
        *temp_storage_bytes,        // temp_storage_bytes (output)
        (const float*)nullptr,      // d_in
        (float*)nullptr,            // d_out
        num_segments,               // num_segments
        (const int*)nullptr,        // d_offsets (begin offsets)
        (const int*)nullptr + 1,    // d_offsets + 1 (end offsets)
        0                           // stream (default)
    );
}

/// Perform segmented sum reduction on f32 data.
///
/// Computes the sum of each segment. Segment i contains elements from
/// d_offsets[i] (inclusive) to d_offsets[i+1] (exclusive).
///
/// # Arguments
/// * `d_temp_storage` - Device temporary storage
/// * `temp_storage_bytes` - Size of temporary storage
/// * `d_in` - Input data (device memory)
/// * `d_out` - Output sums, one per segment (device memory)
/// * `num_segments` - Number of segments
/// * `d_offsets` - Segment offsets array of size num_segments + 1 (device memory)
/// * `stream` - CUDA stream (0 for default stream)
///
/// # Returns
/// cudaSuccess (0) on success, error code otherwise.
CudaError cub_segmented_reduce_sum_f32(
    void* d_temp_storage,
    size_t temp_storage_bytes,
    const float* d_in,
    float* d_out,
    int num_segments,
    const int* d_offsets,
    cudaStream_t stream
) {
    return cub::DeviceSegmentedReduce::Sum(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_segments,
        d_offsets,          // d_begin_offsets
        d_offsets + 1,      // d_end_offsets
        stream
    );
}

/// Query the temporary storage size needed for segmented reduce sum (f64).
///
/// # Arguments
/// * `temp_storage_bytes` - Output: required temporary storage size
/// * `num_items` - Total number of items across all segments
/// * `num_segments` - Number of segments to reduce
///
/// # Returns
/// cudaSuccess (0) on success, error code otherwise.
CudaError cub_segmented_reduce_sum_f64_temp_size(
    size_t* temp_storage_bytes,
    int num_items,
    int num_segments
) {
    return cub::DeviceSegmentedReduce::Sum(
        nullptr,
        *temp_storage_bytes,
        (const double*)nullptr,
        (double*)nullptr,
        num_segments,
        (const int*)nullptr,
        (const int*)nullptr + 1,
        0
    );
}

/// Perform segmented sum reduction on f64 data.
///
/// # Arguments
/// * `d_temp_storage` - Device temporary storage
/// * `temp_storage_bytes` - Size of temporary storage
/// * `d_in` - Input data (device memory)
/// * `d_out` - Output sums, one per segment (device memory)
/// * `num_segments` - Number of segments
/// * `d_offsets` - Segment offsets array of size num_segments + 1 (device memory)
/// * `stream` - CUDA stream (0 for default stream)
///
/// # Returns
/// cudaSuccess (0) on success, error code otherwise.
CudaError cub_segmented_reduce_sum_f64(
    void* d_temp_storage,
    size_t temp_storage_bytes,
    const double* d_in,
    double* d_out,
    int num_segments,
    const int* d_offsets,
    cudaStream_t stream
) {
    return cub::DeviceSegmentedReduce::Sum(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_segments,
        d_offsets,
        d_offsets + 1,
        stream
    );
}

} // extern "C"
