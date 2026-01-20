// Async stream utilities for double-buffered batch processing
//
// Provides C API for:
// - Pinned (page-locked) host memory allocation
// - CUDA stream creation and management
// - CUDA event creation and synchronization
// - Async memory transfers (H2D, D2H)

#include <cuda_runtime.h>
#include <cstdint>

extern "C" {

// ============================================================================
// Pinned Memory Management
// ============================================================================

/// Allocate pinned (page-locked) host memory.
/// Pinned memory enables async transfers and higher bandwidth.
///
/// @param ptr Output pointer to allocated memory
/// @param size Size in bytes to allocate
/// @return CUDA error code
int cuda_malloc_host(void** ptr, size_t size) {
    return cudaMallocHost(ptr, size);
}

/// Free pinned host memory.
///
/// @param ptr Pointer to free
/// @return CUDA error code
int cuda_free_host(void* ptr) {
    return cudaFreeHost(ptr);
}

// ============================================================================
// Stream Management
// ============================================================================

/// Create a new CUDA stream.
///
/// @param stream Output stream handle
/// @return CUDA error code
int cuda_stream_create(cudaStream_t* stream) {
    return cudaStreamCreate(stream);
}

/// Create a new CUDA stream with flags.
///
/// @param stream Output stream handle
/// @param flags Stream creation flags (e.g., cudaStreamNonBlocking)
/// @return CUDA error code
int cuda_stream_create_with_flags(cudaStream_t* stream, unsigned int flags) {
    return cudaStreamCreateWithFlags(stream, flags);
}

/// Destroy a CUDA stream.
///
/// @param stream Stream to destroy
/// @return CUDA error code
int cuda_stream_destroy(cudaStream_t stream) {
    return cudaStreamDestroy(stream);
}

/// Synchronize a stream (block until all operations complete).
///
/// @param stream Stream to synchronize
/// @return CUDA error code
int cuda_stream_synchronize(cudaStream_t stream) {
    return cudaStreamSynchronize(stream);
}

/// Query if stream has completed all operations (non-blocking).
///
/// @param stream Stream to query
/// @return cudaSuccess if complete, cudaErrorNotReady if still running
int cuda_stream_query(cudaStream_t stream) {
    return cudaStreamQuery(stream);
}

// ============================================================================
// Event Management
// ============================================================================

/// Create a new CUDA event.
///
/// @param event Output event handle
/// @return CUDA error code
int cuda_event_create(cudaEvent_t* event) {
    return cudaEventCreate(event);
}

/// Create a new CUDA event with flags.
///
/// @param event Output event handle
/// @param flags Event creation flags (e.g., cudaEventDisableTiming)
/// @return CUDA error code
int cuda_event_create_with_flags(cudaEvent_t* event, unsigned int flags) {
    return cudaEventCreateWithFlags(event, flags);
}

/// Destroy a CUDA event.
///
/// @param event Event to destroy
/// @return CUDA error code
int cuda_event_destroy(cudaEvent_t event) {
    return cudaEventDestroy(event);
}

/// Record an event in a stream.
///
/// @param event Event to record
/// @param stream Stream to record in (0 for default stream)
/// @return CUDA error code
int cuda_event_record(cudaEvent_t event, cudaStream_t stream) {
    return cudaEventRecord(event, stream);
}

/// Query if event has completed (non-blocking).
///
/// @param event Event to query
/// @return cudaSuccess if complete, cudaErrorNotReady if still pending
int cuda_event_query(cudaEvent_t event) {
    return cudaEventQuery(event);
}

/// Synchronize on an event (block until event completes).
///
/// @param event Event to synchronize on
/// @return CUDA error code
int cuda_event_synchronize(cudaEvent_t event) {
    return cudaEventSynchronize(event);
}

/// Compute elapsed time between two events.
///
/// @param ms Output elapsed time in milliseconds
/// @param start Start event
/// @param end End event
/// @return CUDA error code
int cuda_event_elapsed_time(float* ms, cudaEvent_t start, cudaEvent_t end) {
    return cudaEventElapsedTime(ms, start, end);
}

/// Make a stream wait on an event.
///
/// @param stream Stream to make wait
/// @param event Event to wait on
/// @param flags Wait flags (typically 0)
/// @return CUDA error code
int cuda_stream_wait_event(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    return cudaStreamWaitEvent(stream, event, flags);
}

// ============================================================================
// Async Memory Operations
// ============================================================================

/// Async host-to-device memory copy.
///
/// @param dst Device destination pointer
/// @param src Host source pointer
/// @param count Number of bytes to copy
/// @param stream Stream to perform copy in
/// @return CUDA error code
int cuda_memcpy_async_h2d(void* dst, const void* src, size_t count, cudaStream_t stream) {
    return cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream);
}

/// Async device-to-host memory copy.
///
/// @param dst Host destination pointer
/// @param src Device source pointer
/// @param count Number of bytes to copy
/// @param stream Stream to perform copy in
/// @return CUDA error code
int cuda_memcpy_async_d2h(void* dst, const void* src, size_t count, cudaStream_t stream) {
    return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream);
}

/// Async device-to-device memory copy.
///
/// @param dst Device destination pointer
/// @param src Device source pointer
/// @param count Number of bytes to copy
/// @param stream Stream to perform copy in
/// @return CUDA error code
int cuda_memcpy_async_d2d(void* dst, const void* src, size_t count, cudaStream_t stream) {
    return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
}

/// Async memset.
///
/// @param dst Device pointer
/// @param value Value to set (only lowest byte used)
/// @param count Number of bytes to set
/// @param stream Stream to perform operation in
/// @return CUDA error code
int cuda_memset_async(void* dst, int value, size_t count, cudaStream_t stream) {
    return cudaMemsetAsync(dst, value, count, stream);
}

// ============================================================================
// Device Memory Allocation (for convenience)
// ============================================================================

/// Allocate device memory.
///
/// @param ptr Output pointer to allocated memory
/// @param size Size in bytes to allocate
/// @return CUDA error code
int cuda_malloc_device(void** ptr, size_t size) {
    return cudaMalloc(ptr, size);
}

/// Free device memory.
///
/// @param ptr Pointer to free
/// @return CUDA error code
int cuda_free_device(void* ptr) {
    return cudaFree(ptr);
}

// ============================================================================
// Constants
// ============================================================================

/// Get cudaStreamNonBlocking flag value.
unsigned int cuda_stream_non_blocking_flag() {
    return cudaStreamNonBlocking;
}

/// Get cudaEventDisableTiming flag value.
unsigned int cuda_event_disable_timing_flag() {
    return cudaEventDisableTiming;
}

/// Get cudaErrorNotReady value.
int cuda_error_not_ready() {
    return cudaErrorNotReady;
}

} // extern "C"
