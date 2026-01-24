# Phase 24: CUDA Graphs Pipeline

**Status**: 📋 Planned
**Priority**: High
**Motivation**: The cooperative groups persistent kernel (Phase 17) fails on GPUs with limited SM count (Jetson Orin) due to `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE` (error 720).

## Problem Statement

The current persistent NDT kernel uses `cudaLaunchCooperativeKernel` with `cg::grid_group::sync()` for grid-wide barriers. This approach has strict limits on the maximum number of thread blocks:

| GPU | SMs | Max Cooperative Blocks | Points @ 256 threads/block |
|-----|-----|------------------------|---------------------------|
| Jetson Orin | 16 | ~100-200 | 25,600-51,200 |
| RTX 4090 | 128 | ~1,500+ | 384,000+ |
| H100 | 132 | ~2,000+ | 512,000+ |

**Current requirement**: ~1,277 blocks for 326,867 points (typical scan)

**Error**: `CUDA error code 720` on Jetson Orin

## Solution: CUDA Graphs with Kernel Batching

Replace the single cooperative kernel with a **CUDA Graph** that captures multiple smaller kernels, eliminating the cooperative launch limit while maintaining similar performance.

### Key Benefits

1. **No cooperative launch limits** - Each kernel can use standard launch
2. **Reduced launch overhead** - Graph launch is faster than individual kernel launches
3. **Iteration batching** - Batch 4-8 iterations per graph execution for amortization
4. **Portable** - Works on all CUDA GPUs (no SM count dependency)

### Performance Expectations

Based on research ([arxiv:2501.09398](https://arxiv.org/html/2501.09398v1)):

| Approach | vs Multi-Kernel | vs Cooperative Persistent |
|----------|-----------------|---------------------------|
| CUDA Graphs (batched) | 1.4× faster | ~0.9-1.0× (comparable) |
| Naive multi-kernel | 1.0× baseline | ~0.7× (launch overhead) |

## Architecture

### Current Persistent Kernel Phases

The existing kernel has 8 grid synchronization points that serve as natural splitting boundaries:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PERSISTENT KERNEL (single launch)            │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐                                               │
│  │ Initialization│ ←── grid.sync() #1                          │
│  └──────┬───────┘                                               │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    ITERATION LOOP                         │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ Phase A: Per-point score/gradient/Hessian compute  │  │  │
│  │  │          (parallel over all points)                │  │  │
│  │  └────────────────────┬───────────────────────────────┘  │  │
│  │                       ▼                                   │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ Phase B: Block-level reduction + atomic global add │  │  │
│  │  │          ←── grid.sync() #2                        │  │  │
│  │  └────────────────────┬───────────────────────────────┘  │  │
│  │                       ▼                                   │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ Phase C: Newton solve + direction (thread 0 only)  │  │  │
│  │  │          ←── grid.sync() #3                        │  │  │
│  │  └────────────────────┬───────────────────────────────┘  │  │
│  │                       ▼                                   │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ Phase C.2: Line search (if enabled)                │  │  │
│  │  │            ←── grid.sync() #4-7 (batched eval)     │  │  │
│  │  └────────────────────┬───────────────────────────────┘  │  │
│  │                       ▼                                   │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ Phase D: Convergence check                         │  │  │
│  │  │          ←── grid.sync() #8                        │  │  │
│  │  └────────────────────┬───────────────────────────────┘  │  │
│  │                       ▼                                   │  │
│  │                 [loop or exit]                            │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### New CUDA Graph Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                      CUDA GRAPH (captured once)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐                                           │
│  │ K1: Initialization│  (single block, runs once per alignment) │
│  └────────┬─────────┘                                           │
│           ▼                                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              ITERATION BATCH (repeat N times in graph)      │ │
│  │                                                              │ │
│  │  ┌────────────────────────────────────────────────────────┐ │ │
│  │  │ K2: Compute Kernel                                     │ │ │
│  │  │     - Per-point score/gradient/Hessian                 │ │ │
│  │  │     - Block-local reduction to shared memory           │ │ │
│  │  │     - Atomic add to global reduction buffer            │ │ │
│  │  │     Grid: ceil(N/256), Block: 256                      │ │ │
│  │  └────────────────────┬───────────────────────────────────┘ │ │
│  │                       ▼                                      │ │
│  │  ┌────────────────────────────────────────────────────────┐ │ │
│  │  │ K3: Solve Kernel                                       │ │ │
│  │  │     - Read global reduction results                    │ │ │
│  │  │     - Apply regularization                             │ │ │
│  │  │     - Cholesky/SVD solve for Newton direction          │ │ │
│  │  │     - Update pose (or prepare line search)             │ │ │
│  │  │     Grid: 1, Block: 1 (or 32 for warp-level solve)     │ │ │
│  │  └────────────────────┬───────────────────────────────────┘ │ │
│  │                       ▼                                      │ │
│  │  ┌────────────────────────────────────────────────────────┐ │ │
│  │  │ K4: Line Search Kernel (if enabled)                    │ │ │
│  │  │     - Evaluate K candidates in parallel                │ │ │
│  │  │     - Each thread evaluates all candidates for 1 point │ │ │
│  │  │     - Reduce to find best alpha                        │ │ │
│  │  │     Grid: ceil(N/256), Block: 256                      │ │ │
│  │  └────────────────────┬───────────────────────────────────┘ │ │
│  │                       ▼                                      │ │
│  │  ┌────────────────────────────────────────────────────────┐ │ │
│  │  │ K5: Update Kernel                                      │ │ │
│  │  │     - Apply best step to pose                          │ │ │
│  │  │     - Oscillation detection                            │ │ │
│  │  │     - Set convergence flag                             │ │ │
│  │  │     - Clear reduction buffer for next iteration        │ │ │
│  │  │     Grid: 1, Block: 1                                  │ │ │
│  │  └────────────────────────────────────────────────────────┘ │ │
│  │                                                              │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Host loop:
  while (!converged && iter < max_iter) {
      cudaGraphLaunch(graph_exec, stream);
      cudaStreamSynchronize(stream);  // Check convergence flag
      iter += batch_size;
  }
```

## Kernel Specifications

### K1: Initialization Kernel

**Purpose**: Initialize persistent state from initial pose

**Grid**: 1 block × 1 thread
**Shared Memory**: 0

**Inputs**:
- `initial_pose[6]`: Starting pose (x, y, z, roll, pitch, yaw)

**Outputs** (in `state_buffer`):
- `pose[6]`: Current pose (copy of initial)
- `prev_pos[3]`, `prev_prev_pos[3]`: Position history
- `oscillation_count`, `max_oscillation_count`: Counters
- `alpha_sum`: Accumulated step sizes
- `converged`: Flag (0)

**Pseudocode**:
```cuda
__global__ void init_kernel(const float* initial_pose, float* state_buffer) {
    // Copy initial pose
    for (int i = 0; i < 6; i++) state_buffer[POSE_OFFSET + i] = initial_pose[i];
    // Initialize position history
    state_buffer[PREV_POS_OFFSET + 0] = initial_pose[0];
    state_buffer[PREV_POS_OFFSET + 1] = initial_pose[1];
    state_buffer[PREV_POS_OFFSET + 2] = initial_pose[2];
    // ... similar for prev_prev_pos
    // Clear counters
    state_buffer[CONVERGED_OFFSET] = 0.0f;
    state_buffer[OSC_COUNT_OFFSET] = 0.0f;
}
```

---

### K2: Compute Kernel

**Purpose**: Compute per-point NDT score, gradient, and Hessian contributions

**Grid**: `ceil(num_points / 256)` blocks × 256 threads
**Shared Memory**: `256 * 29 * sizeof(float)` = 29 KB (for block reduction)

**Inputs**:
- `source_points[N*3]`: Source point cloud
- `voxel_means[V*3]`: Voxel centroids
- `voxel_inv_covs[V*9]`: Inverse covariance matrices
- `hash_table[capacity]`: Voxel hash table
- `state_buffer.pose[6]`: Current pose

**Outputs**:
- `reduce_buffer[29]`: Accumulated {score, gradient[6], hessian_ut[21], correspondences}

**Algorithm**:
1. Each thread loads one source point
2. Transform point using current pose
3. Hash lookup for neighboring voxels (27-cell search)
4. For each neighbor: accumulate score, gradient, Hessian
5. Block-level tree reduction in shared memory
6. Thread 0 atomically adds to global `reduce_buffer`

**Key optimizations**:
- Use `__ldg()` for read-only data
- Precompute sin/cos for pose rotation
- Unroll neighbor loop (27 iterations)
- Use warp shuffle for final reduction stages

---

### K3: Solve Kernel

**Purpose**: Solve Newton system and compute step direction

**Grid**: 1 block × 32 threads (warp-level parallelism)
**Shared Memory**: 256 bytes (for 6×6 matrix operations)

**Inputs**:
- `reduce_buffer[29]`: Accumulated score, gradient, Hessian
- `state_buffer.pose[6]`: Current pose
- `config`: Regularization parameters, epsilon

**Outputs**:
- `state_buffer.delta[6]`: Newton step direction
- `state_buffer.line_search_state`: If line search enabled

**Algorithm**:
1. Load gradient `g[6]` and Hessian upper triangle `H_ut[21]`
2. Expand `H_ut` to full symmetric `H[6×6]`
3. Apply GNSS regularization (if enabled):
   - Modify score, gradient, Hessian based on reference pose
4. Solve `H * delta = -g`:
   - Try Cholesky decomposition first
   - Fall back to Jacobi SVD if Cholesky fails
5. Validate direction: if `g · delta > 0`, reverse delta
6. If line search disabled: apply fixed step size and update pose
7. If line search enabled: save state for K4

**Warp-level parallelism**:
- Distribute 6×6 matrix operations across 32 threads
- Use warp shuffle for parallel Cholesky/SVD

---

### K4: Line Search Kernel (Optional)

**Purpose**: Evaluate multiple step size candidates in parallel

**Grid**: `ceil(num_points / 256)` blocks × 256 threads
**Shared Memory**: `256 * 8 * 8 * sizeof(float)` = 64 KB (per-candidate reduction)

**Inputs**:
- `source_points[N*3]`: Source point cloud
- `voxel_means[V*3]`, `voxel_inv_covs[V*9]`, `hash_table`: Map data
- `state_buffer.original_pose[6]`: Pose before line search
- `state_buffer.delta[6]`: Newton direction
- `state_buffer.alpha_candidates[8]`: Step size candidates

**Outputs**:
- `state_buffer.candidate_scores[8]`: Score at each candidate
- `state_buffer.candidate_grads[48]`: Gradient at each candidate (for Wolfe check)

**Algorithm**:
1. Each thread evaluates ALL 8 candidates for its assigned point:
   ```
   for (int k = 0; k < 8; k++) {
       trial_pose = original_pose + alpha[k] * delta;
       score_k, grad_k = compute_ndt_contribution(point, trial_pose);
       local_scores[k] += score_k;
       local_grads[k*6:k*6+6] += grad_k;
   }
   ```
2. Block-level reduction for each candidate
3. Atomic add to global per-candidate buffers

**Optimization**: Process candidates in batches of 4 for early termination

---

### K5: Update Kernel

**Purpose**: Apply step, check convergence, prepare for next iteration

**Grid**: 1 block × 1 thread
**Shared Memory**: 0

**Inputs**:
- `state_buffer.*`: All persistent state
- `reduce_buffer[29]`: Current iteration's reduction results
- `config`: Epsilon for convergence

**Outputs**:
- `state_buffer.pose[6]`: Updated pose
- `state_buffer.converged`: Convergence flag
- `state_buffer.iterations`: Iteration counter
- `out_hessian[36]`: Final Hessian (for covariance)

**Algorithm**:
1. If line search enabled:
   - Evaluate Wolfe conditions for each candidate
   - Select best alpha (or fallback to max-score candidate)
   - Apply: `pose = original_pose + best_alpha * delta`
2. Oscillation detection:
   - Compute angle between consecutive movement vectors
   - Increment counter if near-opposite (cos < -0.9)
3. Convergence check:
   - If `step_length < epsilon`: set `converged = 1`
4. Update position history for next iteration
5. Clear `reduce_buffer[0:29]` for next iteration
6. Increment iteration counter

---

## Buffer Layouts

### State Buffer (Persistent Across Iterations)

```
Offset    Size    Field
------    ----    -----
0-5       6       pose[6] - current pose (x, y, z, roll, pitch, yaw)
6-11      6       delta[6] - Newton step direction
12-14     3       prev_pos[3] - previous position (for oscillation)
15-17     3       prev_prev_pos[3] - position before previous
18        1       converged - convergence flag
19        1       iterations - iteration count
20        1       oscillation_count - current oscillation streak
21        1       max_oscillation_count - maximum observed
22        1       alpha_sum - accumulated step sizes
23        1       actual_step_len - step length for convergence check
24-29     6       original_pose[6] - saved for line search
30-37     8       alpha_candidates[8] - line search step sizes
38-45     8       candidate_scores[8] - scores at each candidate
46-93     48      candidate_grads[48] - gradients at each candidate (8×6)
94-101    8       candidate_corr[8] - correspondences at each candidate

Total: 102 floats = 408 bytes
```

### Reduce Buffer (Cleared Each Iteration)

```
Offset    Size    Field
------    ----    -----
0         1       score - accumulated NDT score
1-6       6       gradient[6] - accumulated gradient
7-27      21      hessian_ut[21] - accumulated Hessian upper triangle
28        1       correspondences - point-voxel match count

Total: 29 floats = 116 bytes
```

### Output Buffer

```
Offset    Size    Field
------    ----    -----
0-5       6       final_pose[6]
6         1       iterations
7         1       converged (as float)
8         1       final_score
9-44      36      hessian[36] - full 6×6 for covariance
45        1       num_correspondences
46        1       max_oscillation_count
47        1       avg_alpha (alpha_sum / iterations)

Total: 48 floats = 192 bytes
```

## CUDA Graph Creation

### Graph Structure

```cpp
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

// Create kernel nodes
cudaGraphNode_t init_node, compute_node, solve_node, linesearch_node, update_node;

// K1: Initialization (runs once, not in iteration loop)
cudaKernelNodeParams init_params = {...};
cudaGraphAddKernelNode(&init_node, graph, nullptr, 0, &init_params);

// For iteration batching, unroll N iterations in graph:
cudaGraphNode_t prev_node = init_node;
for (int i = 0; i < BATCH_SIZE; i++) {
    // K2: Compute
    cudaKernelNodeParams compute_params = {...};
    cudaGraphAddKernelNode(&compute_node, graph, &prev_node, 1, &compute_params);

    // K3: Solve
    cudaKernelNodeParams solve_params = {...};
    cudaGraphAddKernelNode(&solve_node, graph, &compute_node, 1, &solve_params);

    // K4: Line Search (optional, conditional)
    if (line_search_enabled) {
        cudaKernelNodeParams ls_params = {...};
        cudaGraphAddKernelNode(&linesearch_node, graph, &solve_node, 1, &ls_params);
        prev_node = linesearch_node;
    } else {
        prev_node = solve_node;
    }

    // K5: Update
    cudaKernelNodeParams update_params = {...};
    cudaGraphAddKernelNode(&update_node, graph, &prev_node, 1, &update_params);

    prev_node = update_node;
}

// Instantiate graph
cudaGraphExec_t graph_exec;
cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
```

### Execution Loop

```cpp
void run_ndt_alignment(/* params */) {
    // Initialize state buffer
    cudaMemcpyAsync(state_buffer, initial_pose, 6*sizeof(float), cudaMemcpyHostToDevice, stream);

    int total_iterations = 0;
    bool converged = false;

    while (!converged && total_iterations < max_iterations) {
        // Launch graph (executes BATCH_SIZE iterations)
        cudaGraphLaunch(graph_exec, stream);

        // Sync and check convergence
        cudaStreamSynchronize(stream);
        cudaMemcpy(&converged, &state_buffer[CONVERGED_OFFSET], sizeof(float), cudaMemcpyDeviceToHost);

        total_iterations += BATCH_SIZE;
    }

    // Copy final results
    cudaMemcpy(output, output_buffer, sizeof(OutputBuffer), cudaMemcpyDeviceToHost);
}
```

## Implementation Roadmap

### Sub-phase 24.1: Kernel Extraction

**Goal**: Extract existing persistent kernel phases into standalone kernels

**Tasks**:
1. Create `ndt_graph_kernels.cu` with separated kernels:
   - `ndt_init_kernel`
   - `ndt_compute_kernel` (Phase A+B from persistent)
   - `ndt_solve_kernel` (Phase C)
   - `ndt_linesearch_kernel` (Phase C.2)
   - `ndt_update_kernel` (Phase C.3+D)

2. Refactor shared device functions to `ndt_graph_device.cuh`:
   - `compute_sincos_inline`, `compute_transform_inline`
   - `hash_query_inline`, `compute_jacobians_inline`
   - Block reduction utilities

3. Define buffer layouts in header:
   - `StateBuffer`, `ReduceBuffer`, `OutputBuffer` structs
   - Offset constants

**Deliverable**: Standalone kernels that can be launched individually

---

### Sub-phase 24.2: CUDA Graph Infrastructure

**Goal**: Create graph capture and execution infrastructure

**Tasks**:
1. Add CUDA Graph FFI bindings to `cuda_ffi`:
   ```rust
   // cuda_ffi/src/graph.rs
   pub fn create_graph() -> Result<CudaGraph, CudaError>;
   pub fn add_kernel_node(...) -> Result<CudaGraphNode, CudaError>;
   pub fn instantiate_graph(graph: &CudaGraph) -> Result<CudaGraphExec, CudaError>;
   pub fn launch_graph(exec: &CudaGraphExec, stream: &CudaStream) -> Result<(), CudaError>;
   ```

2. Create `GraphNdtPipeline` struct in `ndt_cuda`:
   ```rust
   pub struct GraphNdtPipeline {
       graph: CudaGraph,
       graph_exec: CudaGraphExec,
       state_buffer: GpuBuffer,
       reduce_buffer: GpuBuffer,
       output_buffer: GpuBuffer,
       batch_size: usize,
   }
   ```

3. Implement graph construction with iteration batching

**Deliverable**: Rust API for creating and executing NDT graphs

---

### Sub-phase 24.3: Integration with Existing Pipeline

**Goal**: Integrate graph pipeline as alternative to cooperative kernel

**Tasks**:
1. Add runtime selection in `FullGpuPipelineV2`:
   ```rust
   pub enum GpuBackend {
       CooperativeKernel,  // Current implementation
       CudaGraph,          // New graph-based implementation
   }
   ```

2. Auto-detect backend based on GPU capabilities:
   - Query max cooperative blocks
   - If `num_blocks > max_cooperative_blocks`: use Graph
   - Otherwise: use Cooperative (slightly faster)

3. Implement `GraphNdtPipeline::align()` matching existing API

4. Add feature flag: `--features cuda-graph`

**Deliverable**: Drop-in replacement that works on all GPUs

---

### Sub-phase 24.4: Optimization & Benchmarking

**Goal**: Optimize graph performance to match cooperative kernel

**Tasks**:
1. Tune batch size (test 2, 4, 8 iterations per graph)
2. Profile kernel execution times
3. Optimize memory access patterns in separated kernels
4. Consider graph update API for changing parameters

**Benchmarks**:
- Latency per alignment (ms)
- Throughput (alignments/sec)
- Memory bandwidth utilization
- Comparison: Graph vs Cooperative vs Multi-kernel

**Deliverable**: Performance report and optimized implementation

---

### Sub-phase 24.5: Testing & Validation

**Goal**: Ensure numerical equivalence with cooperative kernel

**Tasks**:
1. Unit tests for each kernel
2. Integration test: compare Graph vs Cooperative outputs
3. Rosbag replay validation
4. Edge case testing (early convergence, oscillation, regularization)

**Acceptance Criteria**:
- Pose difference < 1e-6 vs cooperative kernel
- All existing tests pass
- No memory leaks (Valgrind/compute-sanitizer)

**Deliverable**: Validated, production-ready implementation

---

## Timeline Estimate

| Sub-phase | Effort | Dependencies |
|-----------|--------|--------------|
| 24.1 Kernel Extraction | 2-3 days | None |
| 24.2 Graph Infrastructure | 2-3 days | 24.1 |
| 24.3 Integration | 2-3 days | 24.2 |
| 24.4 Optimization | 3-5 days | 24.3 |
| 24.5 Testing | 2-3 days | 24.4 |
| **Total** | **11-17 days** | |

## References

- [CUDA Graphs Getting Started](https://developer.nvidia.com/blog/cuda-graphs/)
- [Boosting Performance of Iterative Applications on GPUs (2025)](https://arxiv.org/html/2501.09398v1)
- [PERKS: Locality-Optimized Execution Model](https://arxiv.org/pdf/2204.02064)
- [Cooperative Groups Blog](https://developer.nvidia.com/blog/cooperative-groups/)

## Appendix: Cooperative vs Graph Comparison

| Aspect | Cooperative Kernel | CUDA Graph |
|--------|-------------------|------------|
| Launch overhead | Single launch | Graph launch (~same) |
| Grid sync | Hardware `grid.sync()` | Implicit between nodes |
| Max blocks | Limited by SM count | Unlimited |
| Data locality | Registers persist | Must use global memory |
| Iteration batching | Automatic (loop inside) | Explicit (unroll in graph) |
| Early exit | Trivial | Requires host check |
| Portability | Requires cooperative support | All CUDA GPUs |
| Complexity | Single kernel | Multiple kernels + graph |
