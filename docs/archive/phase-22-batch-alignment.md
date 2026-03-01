# Phase 22: Batch Multi-Alignment with Non-Cooperative Kernel

## Overview

This phase implements batch processing of multiple NDT alignments in a single kernel launch, replacing the cooperative grid synchronization with atomic barriers to enable true parallel execution across alignment slots.

## Motivation

The current persistent kernel uses cooperative groups (`grid.sync()`) which requires:
- All blocks to participate in synchronization
- Only one cooperative kernel can run at a time
- Entire GPU dedicated to single alignment

This leaves GPU resources underutilized when processing sequential scans. By removing the cooperative requirement and partitioning the GPU into independent slots, we can:
- Process M alignments in parallel
- Reduce kernel launch overhead from O(M) to O(1)
- Reduce memory transfer overhead from O(M) to O(1)
- Achieve near-100% GPU utilization

## Architecture

### Block Partitioning

```
GPU with 80 SMs:
┌─────────────────────────────────────────────────────────────┐
│  Slot 0: Blocks 0-19   (20 blocks, handles alignment 0)     │
│  Slot 1: Blocks 20-39  (20 blocks, handles alignment 1)     │
│  Slot 2: Blocks 40-59  (20 blocks, handles alignment 2)     │
│  Slot 3: Blocks 60-79  (20 blocks, handles alignment 3)     │
└─────────────────────────────────────────────────────────────┘
```

### Memory Layout

```
Shared (read-only):
├── voxel_means[V × 3]
├── voxel_inv_covs[V × 9]
├── hash_table[capacity]
└── Constants: gauss_d1, gauss_d2, resolution

Per-Slot (M copies):
├── source_points[M][max_points × 3]
├── reduce_buffer[M][160]
├── barrier_state[M][2]          # Atomic barrier counters
├── initial_pose[M][6]
├── out_pose[M][6]
├── out_iterations[M]
├── out_converged[M]
├── out_score[M]
├── out_hessian[M][36]
├── out_correspondences[M]
├── out_oscillation[M]
└── out_alpha_sum[M]
```

### Atomic Barrier

Replaces `grid.sync()` with per-slot atomic barrier:

```cuda
__device__ void slot_barrier(
    volatile int* barrier_counter,
    volatile int* barrier_sense,
    int num_blocks_in_slot
) {
    __syncthreads();  // Ensure all threads in block are ready

    if (threadIdx.x == 0) {
        int old_sense = *barrier_sense;

        // Arrive at barrier
        int arrived = atomicAdd((int*)barrier_counter, 1);

        if (arrived == num_blocks_in_slot - 1) {
            // Last block to arrive: reset counter and flip sense
            *barrier_counter = 0;
            __threadfence();  // Ensure counter reset is visible
            atomicAdd((int*)barrier_sense, 1);
        } else {
            // Wait for sense to change
            while (*barrier_sense == old_sense) {
                // Spin
            }
        }
    }

    __syncthreads();  // Ensure all threads see barrier completion
}
```

### Kernel Structure

```cuda
__global__ void batch_persistent_ndt_kernel(
    // Shared data (read-only)
    const float* voxel_means,
    const float* voxel_inv_covs,
    const HashEntry* hash_table,
    uint32_t hash_capacity,
    float gauss_d1, float gauss_d2, float resolution,

    // Per-slot data (M alignments)
    const float* source_points,      // [M * max_points * 3]
    float* reduce_buffers,           // [M * 160]
    int* barrier_counters,           // [M]
    int* barrier_senses,             // [M]
    const float* initial_poses,      // [M * 6]
    float* out_poses,                // [M * 6]
    int* out_iterations,             // [M]
    uint32_t* out_converged,         // [M]
    float* out_scores,               // [M]
    float* out_hessians,             // [M * 36]
    uint32_t* out_correspondences,   // [M]
    uint32_t* out_oscillations,      // [M]
    float* out_alpha_sums,           // [M]

    // Control parameters
    int num_slots,
    int blocks_per_slot,
    const int* points_per_slot,      // [M]
    int max_points_per_slot,
    int max_iterations,
    float epsilon_sq,

    // Line search parameters
    int ls_enabled,
    float ls_mu, float ls_nu,
    float fixed_step_size,

    // Regularization parameters (per-slot)
    const float* reg_ref_x,          // [M]
    const float* reg_ref_y,          // [M]
    float reg_scale,
    int reg_enabled
)
```

## Implementation Plan

### Phase 22.1: Atomic Barrier Infrastructure

**Files:**
- `src/cuda_ffi/csrc/batch_persistent_ndt.cu` (new)
- `src/cuda_ffi/src/batch_persistent_ndt.rs` (new)

**Tasks:**
1. Create new kernel file with atomic barrier implementation
2. Add slot-aware block indexing
3. Add per-slot reduce buffer management
4. Implement basic Newton iteration with slot barriers
5. Add FFI bindings for batch kernel launch

**Validation:**
- Unit test: barrier correctness with varying block counts
- Unit test: multiple slots converge independently

### Phase 22.2: Batch Memory Management

**Files:**
- `src/ndt_cuda/src/optimization/batch_pipeline.rs` (new)
- `src/ndt_cuda/src/optimization/mod.rs` (update)

**Tasks:**
1. Create `BatchGpuPipeline` struct with M-slot buffers
2. Implement batched upload (single memcpy for M scans)
3. Implement batched download (single memcpy for M results)
4. Add slot allocation/deallocation logic

**Validation:**
- Unit test: upload/download M alignments
- Benchmark: verify O(1) transfer overhead

### Phase 22.3: Line Search Integration

**Files:**
- `src/cuda_ffi/csrc/batch_persistent_ndt.cu` (update)

**Tasks:**
1. Port batched parallel line search to batch kernel
2. Add per-slot line search state in reduce buffer
3. Verify early termination works per-slot

**Validation:**
- Unit test: line search converges independently per slot
- Compare results with single-alignment kernel

### Phase 22.4: High-Level API

**Files:**
- `src/ndt_cuda/src/ndt.rs` (update)
- `src/ndt_cuda/src/optimization/solver.rs` (update)

**Tasks:**
1. Add `align_batch_parallel()` method to `NdtScanMatcher`
2. Update `NdtOptimizer` to use batch pipeline
3. Add configuration for slot count

**Validation:**
- Integration test: batch vs sequential results match
- Benchmark: throughput improvement

### Phase 22.5: ROS Integration with Real-Time Scan Queue

**Files:**
- `src/cuda_ndt_matcher/src/scan_queue.rs` (new)
- `src/cuda_ndt_matcher/src/main.rs` (update)
- `src/cuda_ndt_matcher/src/ndt_manager.rs` (update)

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ROS Node Architecture                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PointCloud2 ──► [Scan Callback] ──► [ScanQueue] ──► [Batch Processor]  │
│                         │                 │                 │            │
│                         ▼                 ▼                 ▼            │
│                   - Downsample      - Max queue: 8    - align_parallel   │
│                   - Get EKF pose    - Drop oldest     - Async publish    │
│                   - Enqueue         - Real-time       - Per-scan result  │
│                                       constraint                         │
│                                                                          │
│  Output: PoseWithCovariance, TF, Diagnostics (per-scan, in order)       │
└─────────────────────────────────────────────────────────────────────────┘
```

**ScanQueue Design:**

```rust
/// Queued scan request awaiting batch processing.
pub struct QueuedScan {
    /// Source point cloud (downsampled)
    pub points: Vec<[f32; 3]>,
    /// Initial pose from EKF
    pub initial_pose: Isometry3<f64>,
    /// Original message timestamp for output correlation
    pub timestamp: Time,
    /// Arrival time for latency tracking
    pub arrival_ns: u64,
}

/// Real-time scan queue with deadline-based dropping.
pub struct ScanQueue {
    /// Pending scans waiting for batch processing
    queue: VecDeque<QueuedScan>,
    /// Maximum queue depth (default: 8)
    max_depth: usize,
    /// Maximum scan age before dropping (default: 100ms)
    max_age_ms: u64,
    /// Batch processing trigger threshold (default: 4)
    batch_trigger: usize,
    /// Processing thread handle
    processor: Option<JoinHandle<()>>,
    /// Channel to send scans to processor
    tx: Sender<QueuedScan>,
}
```

**Real-Time Constraints:**

1. **Maximum Queue Depth** (default: 8)
   - When queue is full, drop oldest scan
   - Prevents unbounded memory growth
   - Allows catching up after processing stalls

2. **Maximum Scan Age** (default: 100ms)
   - Drop scans older than threshold
   - Prevents processing stale data
   - Maintains real-time responsiveness

3. **Batch Trigger Threshold** (default: 4)
   - Process batch when N scans accumulated
   - Balance latency vs throughput
   - Configurable via parameter

4. **Timeout Trigger** (default: 20ms)
   - Process partial batch if no new scans arrive
   - Prevents indefinite waiting
   - Ensures minimum latency bound

**Processing Flow:**

```
1. Scan arrives via ROS callback
   ├─ Downsample points
   ├─ Get EKF pose as initial guess
   └─ Enqueue (timestamp, points, pose)

2. Queue management (in enqueue)
   ├─ If queue full: drop oldest scan
   ├─ If scan too old: drop immediately
   └─ If batch_trigger reached: wake processor

3. Batch processor thread (async)
   ├─ Wait for batch_trigger scans OR timeout
   ├─ Dequeue up to 8 scans
   ├─ Call matcher.align_parallel_scans()
   └─ For each result: publish (pose, TF, diagnostics)

4. Result publishing (preserves ordering)
   ├─ Sort results by original timestamp
   └─ Publish in timestamp order
```

**Configuration Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch.max_queue_depth` | 8 | Maximum scans in queue |
| `batch.max_scan_age_ms` | 100 | Drop scans older than this |
| `batch.trigger_threshold` | 4 | Process when N scans ready |
| `batch.timeout_ms` | 20 | Process partial batch after timeout |
| `batch.enabled` | true | Enable batch processing |

**Validation:**
- End-to-end test with rosbag replay
- Verify pose output matches single-alignment mode
- Measure throughput improvement (target: 3x+)
- Verify latency bounds are respected
- Test queue overflow behavior

## Buffer Sizing

### Per-Slot Requirements

| Buffer | Size per Slot | Notes |
|--------|--------------|-------|
| source_points | N × 12 bytes | N = max points per scan |
| reduce_buffer | 640 bytes | 160 floats |
| barrier_state | 8 bytes | 2 ints |
| initial_pose | 24 bytes | 6 floats |
| out_pose | 24 bytes | 6 floats |
| out_iterations | 4 bytes | 1 int |
| out_converged | 4 bytes | 1 uint |
| out_score | 4 bytes | 1 float |
| out_hessian | 144 bytes | 36 floats |
| out_correspondences | 4 bytes | 1 uint |
| out_oscillation | 4 bytes | 1 uint |
| out_alpha_sum | 4 bytes | 1 float |
| **Total (excl. points)** | **864 bytes** | |

### Example Configuration

For M=4 slots, N=2000 points max:
- Shared voxel data: ~12 MB (100k voxels)
- Per-slot points: 4 × 2000 × 12 = 96 KB
- Per-slot state: 4 × 864 = 3.5 KB
- **Total additional**: ~100 KB

GPU memory is not a constraint.

## Performance Expectations

### Throughput

| Metric | Single Alignment | Batch (M=4) | Improvement |
|--------|-----------------|-------------|-------------|
| Kernel launches | M | 1 | 4x fewer |
| Upload operations | M | 1 | 4x fewer |
| Download operations | M | 1 | 4x fewer |
| Total latency (4 scans) | 4 × 5ms = 20ms | ~7ms | 2.8x faster |

### Latency Breakdown

```
Single alignment:
  Upload: 0.5ms
  Kernel: 5.0ms
  Download: 0.1ms
  Total: 5.6ms × 4 = 22.4ms for 4 alignments

Batch (M=4):
  Upload (4 scans): 0.8ms
  Kernel (4 parallel): 6.0ms (overhead from barriers)
  Download (4 results): 0.2ms
  Total: 7.0ms for 4 alignments

Speedup: 3.2x
```

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Atomic barrier contention | Tune blocks_per_slot to minimize contention |
| Uneven convergence (some slots finish early) | Early-finished slots spin-wait; acceptable overhead |
| Memory bandwidth saturation | Shared voxel data is read-only, cached |
| Debug complexity | Add per-slot debug output buffer |

## Success Criteria

1. **Correctness**: Batch results match single-alignment results (< 1e-6 error)
2. **Throughput**: 2.5x+ improvement for M=4 batch
3. **Latency**: < 8ms for 4 concurrent alignments
4. **Stability**: No deadlocks or race conditions in 10,000+ batch runs

## Future Extensions

### Phase 22.6: Persistent Daemon Kernel (Optional)

Run kernel indefinitely with GPU-side work queue:
- Zero kernel launch overhead
- CPU just sets flags in mapped memory
- Maximum possible throughput

### Phase 22.7: Dynamic Slot Sizing (Optional)

Adjust blocks_per_slot based on point counts:
- Scans with more points get more blocks
- Better load balancing across slots
