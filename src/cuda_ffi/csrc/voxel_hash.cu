// Spatial hash table for GPU-accelerated voxel lookup
//
// This implements a hash table mapping 3D grid coordinates to voxel indices.
// Key features:
// - Open addressing with linear probing for GPU efficiency
// - Pre-computed neighbor offsets for 27-cell (3x3x3) queries
// - O(27) lookups per query instead of O(V) brute-force
//
// Hash function: Based on spatial hashing with prime number mixing
// to minimize collisions for 3D grid coordinates.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

extern "C" {

// Error codes
typedef int CudaError;

// Empty slot marker
constexpr int32_t EMPTY_SLOT = -1;

// Hash table entry: stores grid coordinate (packed) and voxel index
struct HashEntry {
    int64_t key;     // Packed (x, y, z) grid coordinate, -1 for empty
    int32_t value;   // Voxel index
    int32_t padding; // Alignment padding
};

// Pack 3 int32 grid coordinates into int64 key
// Using 21 bits per coordinate, supports range [-1048576, 1048575]
__device__ __host__ inline int64_t pack_key(int32_t gx, int32_t gy, int32_t gz) {
    // Shift to unsigned range [0, 2^21) then pack
    int64_t ux = (int64_t)(gx + (1 << 20));
    int64_t uy = (int64_t)(gy + (1 << 20));
    int64_t uz = (int64_t)(gz + (1 << 20));
    return (ux << 42) | (uy << 21) | uz;
}

// Unpack int64 key to 3 int32 grid coordinates
__device__ __host__ inline void unpack_key(int64_t key, int32_t* gx, int32_t* gy, int32_t* gz) {
    *gz = (int32_t)((key & 0x1FFFFF) - (1 << 20));
    *gy = (int32_t)(((key >> 21) & 0x1FFFFF) - (1 << 20));
    *gx = (int32_t)(((key >> 42) & 0x1FFFFF) - (1 << 20));
}

// Hash function for 64-bit key
// Based on MurmurHash3 finalizer
__device__ __host__ inline uint32_t hash_key(int64_t key, uint32_t capacity) {
    uint64_t k = (uint64_t)key;
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return (uint32_t)(k % capacity);
}

// Convert world position to grid coordinate
__device__ inline int32_t pos_to_grid(float pos, float inv_resolution) {
    return (int32_t)floorf(pos * inv_resolution);
}

// ============================================================================
// Build kernel: Insert voxels into hash table
// ============================================================================

__global__ void voxel_hash_build_kernel(
    const float* __restrict__ voxel_means,  // [V * 3]
    const uint32_t* __restrict__ voxel_valid, // [V]
    uint32_t num_voxels,
    float inv_resolution,
    HashEntry* __restrict__ hash_table,
    uint32_t capacity
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_voxels) return;

    // Skip invalid voxels
    if (voxel_valid[idx] == 0) return;

    // Get voxel mean position
    float mx = voxel_means[idx * 3 + 0];
    float my = voxel_means[idx * 3 + 1];
    float mz = voxel_means[idx * 3 + 2];

    // Convert to grid coordinates
    int32_t gx = pos_to_grid(mx, inv_resolution);
    int32_t gy = pos_to_grid(my, inv_resolution);
    int32_t gz = pos_to_grid(mz, inv_resolution);

    // Pack key
    int64_t key = pack_key(gx, gy, gz);

    // Insert with linear probing
    uint32_t slot = hash_key(key, capacity);
    for (uint32_t i = 0; i < capacity; i++) {
        uint32_t probe_slot = (slot + i) % capacity;

        // Try to claim this slot
        int64_t old = atomicCAS((unsigned long long*)&hash_table[probe_slot].key,
                                (unsigned long long)EMPTY_SLOT,
                                (unsigned long long)key);

        if (old == EMPTY_SLOT || old == key) {
            // Successfully claimed or found existing entry
            // Store the voxel index (last write wins for duplicates)
            hash_table[probe_slot].value = (int32_t)idx;
            return;
        }
    }
    // Table is full - this shouldn't happen with proper capacity
}

// ============================================================================
// Query kernel: Find neighbors for each query point
// ============================================================================

// Maximum neighbors to return per query point
constexpr uint32_t MAX_NEIGHBORS = 8;

// 27 neighbor offsets (3x3x3 cube)
__constant__ int8_t NEIGHBOR_OFFSETS[27][3] = {
    {-1, -1, -1}, {-1, -1,  0}, {-1, -1,  1},
    {-1,  0, -1}, {-1,  0,  0}, {-1,  0,  1},
    {-1,  1, -1}, {-1,  1,  0}, {-1,  1,  1},
    { 0, -1, -1}, { 0, -1,  0}, { 0, -1,  1},
    { 0,  0, -1}, { 0,  0,  0}, { 0,  0,  1},
    { 0,  1, -1}, { 0,  1,  0}, { 0,  1,  1},
    { 1, -1, -1}, { 1, -1,  0}, { 1, -1,  1},
    { 1,  0, -1}, { 1,  0,  0}, { 1,  0,  1},
    { 1,  1, -1}, { 1,  1,  0}, { 1,  1,  1}
};

__global__ void voxel_hash_query_kernel(
    const float* __restrict__ query_points,   // [N * 3] transformed points
    const float* __restrict__ voxel_means,    // [V * 3] voxel means for distance check
    uint32_t num_queries,
    float inv_resolution,
    float radius_sq,                          // Search radius squared
    const HashEntry* __restrict__ hash_table,
    uint32_t capacity,
    int32_t* __restrict__ neighbor_indices,   // [N * MAX_NEIGHBORS] output
    uint32_t* __restrict__ neighbor_counts    // [N] output
) {
    uint32_t query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_queries) return;

    // Load query point
    float qx = query_points[query_idx * 3 + 0];
    float qy = query_points[query_idx * 3 + 1];
    float qz = query_points[query_idx * 3 + 2];

    // Convert to grid coordinates
    int32_t gx = pos_to_grid(qx, inv_resolution);
    int32_t gy = pos_to_grid(qy, inv_resolution);
    int32_t gz = pos_to_grid(qz, inv_resolution);

    // Initialize output
    uint32_t out_base = query_idx * MAX_NEIGHBORS;
    for (uint32_t i = 0; i < MAX_NEIGHBORS; i++) {
        neighbor_indices[out_base + i] = -1;
    }

    uint32_t count = 0;

    // Check all 27 neighboring cells
    for (uint32_t n = 0; n < 27 && count < MAX_NEIGHBORS; n++) {
        int32_t nx = gx + NEIGHBOR_OFFSETS[n][0];
        int32_t ny = gy + NEIGHBOR_OFFSETS[n][1];
        int32_t nz = gz + NEIGHBOR_OFFSETS[n][2];

        int64_t key = pack_key(nx, ny, nz);
        uint32_t slot = hash_key(key, capacity);

        // Linear probing search
        for (uint32_t i = 0; i < capacity && count < MAX_NEIGHBORS; i++) {
            uint32_t probe_slot = (slot + i) % capacity;
            int64_t stored_key = hash_table[probe_slot].key;

            if (stored_key == EMPTY_SLOT) {
                // Empty slot means key not found
                break;
            }

            if (stored_key == key) {
                // Found matching cell - check distance
                int32_t voxel_idx = hash_table[probe_slot].value;

                float vx = voxel_means[voxel_idx * 3 + 0];
                float vy = voxel_means[voxel_idx * 3 + 1];
                float vz = voxel_means[voxel_idx * 3 + 2];

                float dx = qx - vx;
                float dy = qy - vy;
                float dz = qz - vz;
                float dist_sq = dx * dx + dy * dy + dz * dz;

                if (dist_sq <= radius_sq) {
                    neighbor_indices[out_base + count] = voxel_idx;
                    count++;
                }
                break; // Found the cell, no need to continue probing
            }
        }
    }

    neighbor_counts[query_idx] = count;
}

// ============================================================================
// Host API
// ============================================================================

/// Query required hash table capacity for given number of voxels.
/// Returns capacity with ~50% load factor for good performance.
CudaError voxel_hash_get_capacity(
    uint32_t num_voxels,
    uint32_t* capacity
) {
    // Use ~50% load factor (2x voxels) and round up to power of 2
    uint32_t min_cap = num_voxels * 2;
    uint32_t cap = 1;
    while (cap < min_cap) cap *= 2;
    *capacity = cap;
    return cudaSuccess;
}

/// Query required memory size for hash table.
CudaError voxel_hash_get_table_size(
    uint32_t capacity,
    size_t* bytes
) {
    *bytes = capacity * sizeof(HashEntry);
    return cudaSuccess;
}

/// Initialize hash table (set all entries to empty).
CudaError voxel_hash_init(
    void* d_hash_table,
    uint32_t capacity,
    cudaStream_t stream
) {
    return cudaMemsetAsync(d_hash_table, 0xFF, capacity * sizeof(HashEntry), stream);
}

/// Build hash table from voxel means.
///
/// # Arguments
/// * `d_voxel_means` - Device pointer to voxel means [V * 3]
/// * `d_voxel_valid` - Device pointer to voxel validity flags [V]
/// * `num_voxels` - Number of voxels
/// * `resolution` - Voxel grid resolution (e.g., 2.0)
/// * `d_hash_table` - Device pointer to hash table (must be initialized)
/// * `capacity` - Hash table capacity
/// * `stream` - CUDA stream
CudaError voxel_hash_build(
    const float* d_voxel_means,
    const uint32_t* d_voxel_valid,
    uint32_t num_voxels,
    float resolution,
    void* d_hash_table,
    uint32_t capacity,
    cudaStream_t stream
) {
    if (num_voxels == 0) return cudaSuccess;

    float inv_resolution = 1.0f / resolution;

    int threads = 256;
    int blocks = (num_voxels + threads - 1) / threads;

    voxel_hash_build_kernel<<<blocks, threads, 0, stream>>>(
        d_voxel_means,
        d_voxel_valid,
        num_voxels,
        inv_resolution,
        (HashEntry*)d_hash_table,
        capacity
    );

    return cudaGetLastError();
}

/// Query neighbors for multiple points using hash table.
///
/// # Arguments
/// * `d_query_points` - Device pointer to query points [N * 3]
/// * `d_voxel_means` - Device pointer to voxel means [V * 3]
/// * `num_queries` - Number of query points
/// * `resolution` - Voxel grid resolution
/// * `search_radius` - Search radius (typically = resolution)
/// * `d_hash_table` - Device pointer to hash table
/// * `capacity` - Hash table capacity
/// * `d_neighbor_indices` - Device pointer to output indices [N * MAX_NEIGHBORS]
/// * `d_neighbor_counts` - Device pointer to output counts [N]
/// * `stream` - CUDA stream
CudaError voxel_hash_query(
    const float* d_query_points,
    const float* d_voxel_means,
    uint32_t num_queries,
    float resolution,
    float search_radius,
    const void* d_hash_table,
    uint32_t capacity,
    int32_t* d_neighbor_indices,
    uint32_t* d_neighbor_counts,
    cudaStream_t stream
) {
    if (num_queries == 0) return cudaSuccess;

    float inv_resolution = 1.0f / resolution;
    float radius_sq = search_radius * search_radius;

    int threads = 256;
    int blocks = (num_queries + threads - 1) / threads;

    voxel_hash_query_kernel<<<blocks, threads, 0, stream>>>(
        d_query_points,
        d_voxel_means,
        num_queries,
        inv_resolution,
        radius_sq,
        (const HashEntry*)d_hash_table,
        capacity,
        d_neighbor_indices,
        d_neighbor_counts
    );

    return cudaGetLastError();
}

/// Get MAX_NEIGHBORS constant for Rust binding.
uint32_t voxel_hash_max_neighbors() {
    return MAX_NEIGHBORS;
}

} // extern "C"
