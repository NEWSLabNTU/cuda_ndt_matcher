// Texture memory support for voxel data
//
// Provides C API for:
// - Creating texture objects for voxel means and inverse covariances
// - Destroying texture objects
// - Querying texture object validity
//
// Texture memory provides:
// - Separate texture cache (doesn't compete with L1/L2)
// - Hardware-accelerated caching for scattered reads
// - Better performance for read-only data with spatial locality

#include <cuda_runtime.h>
#include <cstdint>

extern "C" {

// ============================================================================
// Texture Object Creation
// ============================================================================

/// Create a texture object for voxel means array.
///
/// @param tex_out Output texture object handle
/// @param d_means Device pointer to voxel means [num_voxels * 3] floats
/// @param num_voxels Number of voxels
/// @return CUDA error code
int create_voxel_means_texture(
    cudaTextureObject_t* tex_out,
    const float* d_means,
    size_t num_voxels
) {
    // Resource descriptor - describes the data
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = const_cast<float*>(d_means);
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32;  // 32-bit float
    resDesc.res.linear.desc.y = 0;
    resDesc.res.linear.desc.z = 0;
    resDesc.res.linear.desc.w = 0;
    resDesc.res.linear.sizeInBytes = num_voxels * 3 * sizeof(float);

    // Texture descriptor - describes how to sample
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;  // No interpolation (exact values)
    texDesc.readMode = cudaReadModeElementType;  // Return raw float values
    texDesc.normalizedCoords = 0;  // Use integer indices

    return cudaCreateTextureObject(tex_out, &resDesc, &texDesc, nullptr);
}

/// Create a texture object for voxel inverse covariances array.
///
/// @param tex_out Output texture object handle
/// @param d_inv_covs Device pointer to inverse covariances [num_voxels * 9] floats
/// @param num_voxels Number of voxels
/// @return CUDA error code
int create_voxel_inv_covs_texture(
    cudaTextureObject_t* tex_out,
    const float* d_inv_covs,
    size_t num_voxels
) {
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = const_cast<float*>(d_inv_covs);
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32;
    resDesc.res.linear.desc.y = 0;
    resDesc.res.linear.desc.z = 0;
    resDesc.res.linear.desc.w = 0;
    resDesc.res.linear.sizeInBytes = num_voxels * 9 * sizeof(float);

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    return cudaCreateTextureObject(tex_out, &resDesc, &texDesc, nullptr);
}

/// Destroy a texture object.
///
/// @param tex Texture object to destroy
/// @return CUDA error code
int destroy_texture_object(cudaTextureObject_t tex) {
    if (tex == 0) {
        return cudaSuccess;  // Null texture is a no-op
    }
    return cudaDestroyTextureObject(tex);
}

/// Get the size of cudaTextureObject_t for FFI.
///
/// @return Size in bytes
size_t texture_object_size() {
    return sizeof(cudaTextureObject_t);
}

// ============================================================================
// Texture-enabled kernel launch wrapper
// ============================================================================

// Forward declaration of the texture-enabled batch kernel
// (will be implemented in batch_persistent_ndt.cu)
extern int batch_persistent_ndt_launch_with_textures(
    // Texture objects (instead of raw pointers)
    cudaTextureObject_t tex_voxel_means,
    cudaTextureObject_t tex_voxel_inv_covs,

    // Hash table (still raw pointer - not worth texturing)
    const void* hash_table,
    uint32_t hash_capacity,
    float gauss_d1,
    float gauss_d2,
    float resolution,

    // Per-slot input (same as before)
    const float* all_source_points,
    const float* all_initial_poses,
    const int* points_per_slot,

    // Per-slot working memory
    float* all_reduce_buffers,
    int* barrier_counters,
    int* barrier_senses,

    // Per-slot outputs
    float* all_out_poses,
    int* all_out_iterations,
    uint32_t* all_out_converged,
    float* all_out_scores,
    float* all_out_hessians,
    uint32_t* all_out_correspondences,
    uint32_t* all_out_oscillations,
    float* all_out_alpha_sums,

    // Control
    int num_slots,
    int blocks_per_slot,
    int max_points_per_slot,
    int max_iterations,
    float epsilon,

    // Line search
    int ls_enabled,
    int ls_num_candidates,
    float ls_mu,
    float ls_nu,
    float fixed_step_size,

    // Regularization
    const float* reg_ref_x,
    const float* reg_ref_y,
    float reg_scale,
    int reg_enabled,

    // Stream
    cudaStream_t stream
);

/// Launch batch persistent NDT kernel with texture memory for voxel data.
///
/// This is the texture-enabled version that uses texture cache for
/// voxel means and inverse covariances.
int batch_persistent_ndt_launch_textured(
    // Texture objects
    cudaTextureObject_t tex_voxel_means,
    cudaTextureObject_t tex_voxel_inv_covs,

    // Hash table
    const void* hash_table,
    uint32_t hash_capacity,
    float gauss_d1,
    float gauss_d2,
    float resolution,

    // Per-slot input
    const float* all_source_points,
    const float* all_initial_poses,
    const int* points_per_slot,

    // Per-slot working memory
    float* all_reduce_buffers,
    int* barrier_counters,
    int* barrier_senses,

    // Per-slot outputs
    float* all_out_poses,
    int* all_out_iterations,
    uint32_t* all_out_converged,
    float* all_out_scores,
    float* all_out_hessians,
    uint32_t* all_out_correspondences,
    uint32_t* all_out_oscillations,
    float* all_out_alpha_sums,

    // Control
    int num_slots,
    int blocks_per_slot,
    int max_points_per_slot,
    int max_iterations,
    float epsilon,

    // Line search
    int ls_enabled,
    int ls_num_candidates,
    float ls_mu,
    float ls_nu,
    float fixed_step_size,

    // Regularization
    const float* reg_ref_x,
    const float* reg_ref_y,
    float reg_scale,
    int reg_enabled,

    // Stream
    cudaStream_t stream
) {
    return batch_persistent_ndt_launch_with_textures(
        tex_voxel_means, tex_voxel_inv_covs,
        hash_table, hash_capacity,
        gauss_d1, gauss_d2, resolution,
        all_source_points, all_initial_poses, points_per_slot,
        all_reduce_buffers, barrier_counters, barrier_senses,
        all_out_poses, all_out_iterations, all_out_converged,
        all_out_scores, all_out_hessians,
        all_out_correspondences, all_out_oscillations, all_out_alpha_sums,
        num_slots, blocks_per_slot, max_points_per_slot,
        max_iterations, epsilon,
        ls_enabled, ls_num_candidates, ls_mu, ls_nu, fixed_step_size,
        reg_ref_x, reg_ref_y, reg_scale, reg_enabled,
        stream
    );
}

} // extern "C"
