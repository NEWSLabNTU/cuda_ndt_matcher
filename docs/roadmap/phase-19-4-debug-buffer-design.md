# Phase 19.4: Per-Iteration Debug Buffer Design

This document describes the design for collecting per-iteration debug data from the persistent kernel with zero overhead when disabled.

## Requirements

1. **Zero overhead when disabled** - Default case should have negligible performance impact
2. **Complete iteration history** - Capture all fields needed for `IterationDebug` reconstruction
3. **Single kernel launch** - No additional kernel launches or synchronization points
4. **Minimal memory footprint** - Reasonable buffer size for typical max_iterations (30)

## Buffer Layout

### Per-Iteration Data (50 floats = 200 bytes)

| Offset | Count | Field | Description |
|--------|-------|-------|-------------|
| 0 | 1 | iteration | Iteration number (0-indexed) |
| 1 | 1 | score | NDT score at current pose |
| 2-7 | 6 | pose_before | Pose at start of iteration [tx,ty,tz,r,p,y] |
| 8-13 | 6 | gradient | Gradient vector |
| 14-34 | 21 | hessian_ut | Hessian upper triangle (row-major) |
| 35-40 | 6 | delta | Newton step (H⁻¹g) |
| 41 | 1 | alpha | Step size from line search (or 1.0) |
| 42 | 1 | correspondences | Number of point-voxel correspondences |
| 43 | 1 | direction_reversed | 1.0 if direction reversed, 0.0 otherwise |
| 44-49 | 6 | pose_after | Pose after step applied |

**Total buffer size:** 50 floats × max_iterations = 1500 floats for 30 iterations = 6000 bytes

### Hessian Upper Triangle Mapping

The 6×6 symmetric Hessian is stored as 21 upper triangle elements:

```
H[0,0] H[0,1] H[0,2] H[0,3] H[0,4] H[0,5]    →  [0]  [1]  [2]  [3]  [4]  [5]
       H[1,1] H[1,2] H[1,3] H[1,4] H[1,5]    →       [6]  [7]  [8]  [9] [10]
              H[2,2] H[2,3] H[2,4] H[2,5]    →           [11] [12] [13] [14]
                     H[3,3] H[3,4] H[3,5]    →                [15] [16] [17]
                            H[4,4] H[4,5]    →                     [18] [19]
                                   H[5,5]    →                          [20]
```

## CUDA Kernel Changes

### New Parameters

```c
__global__ void persistent_ndt_kernel(
    // ... existing parameters ...

    // Debug output (Phase 19.4)
    int32_t debug_enabled,           // 0 = disabled, 1 = enabled
    float* __restrict__ debug_buffer // [max_iterations * 50] or nullptr
) {
```

### Debug Constants

```c
constexpr int DEBUG_FLOATS_PER_ITER = 50;
constexpr int DEBUG_OFF_ITERATION = 0;
constexpr int DEBUG_OFF_SCORE = 1;
constexpr int DEBUG_OFF_POSE_BEFORE = 2;   // 6 floats
constexpr int DEBUG_OFF_GRADIENT = 8;      // 6 floats
constexpr int DEBUG_OFF_HESSIAN_UT = 14;   // 21 floats
constexpr int DEBUG_OFF_DELTA = 35;        // 6 floats
constexpr int DEBUG_OFF_ALPHA = 41;
constexpr int DEBUG_OFF_CORRESPONDENCES = 42;
constexpr int DEBUG_OFF_REVERSED = 43;
constexpr int DEBUG_OFF_POSE_AFTER = 44;   // 6 floats
```

### Write Logic (at end of each iteration)

```c
// Phase 19.4: Write debug data (only if enabled)
if (debug_enabled && threadIdx.x == 0 && blockIdx.x == 0) {
    float* iter_debug = &debug_buffer[iter * DEBUG_FLOATS_PER_ITER];

    // Iteration number
    iter_debug[DEBUG_OFF_ITERATION] = (float)iter;

    // Score
    iter_debug[DEBUG_OFF_SCORE] = *g_final_score;

    // Pose before step (saved at iteration start)
    for (int i = 0; i < 6; i++) {
        iter_debug[DEBUG_OFF_POSE_BEFORE + i] = g_pose_before[i];
    }

    // Gradient (from reduce_buffer)
    for (int i = 0; i < 6; i++) {
        iter_debug[DEBUG_OFF_GRADIENT + i] = reduce_buffer[1 + i];
    }

    // Hessian upper triangle (from reduce_buffer)
    for (int i = 0; i < 21; i++) {
        iter_debug[DEBUG_OFF_HESSIAN_UT + i] = reduce_buffer[7 + i];
    }

    // Delta (Newton step)
    for (int i = 0; i < 6; i++) {
        iter_debug[DEBUG_OFF_DELTA + i] = g_delta[i];
    }

    // Alpha and correspondences
    iter_debug[DEBUG_OFF_ALPHA] = ls_enabled ? *g_best_alpha : 1.0f;
    iter_debug[DEBUG_OFF_CORRESPONDENCES] = *g_total_corr;

    // Direction reversed (detected during line search or Newton solve)
    iter_debug[DEBUG_OFF_REVERSED] = *g_direction_reversed;

    // Pose after step
    for (int i = 0; i < 6; i++) {
        iter_debug[DEBUG_OFF_POSE_AFTER + i] = g_pose[i];
    }
}
```

### Additional State Tracking

Need to add to reduce_buffer or local state:
- `g_pose_before[6]` - Save pose at start of iteration
- `g_direction_reversed` - Track if Newton direction was reversed

```c
// At start of iteration (before pose update)
if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int i = 0; i < 6; i++) {
        g_pose_before[i] = g_pose[i];
    }
}

// After Newton solve, check if direction was reversed
// (gradient · delta > 0 means ascent direction, which is correct for maximization)
float gd = 0.0f;
for (int i = 0; i < 6; i++) {
    gd += reduce_buffer[1 + i] * g_delta[i];
}
*g_direction_reversed = (gd < 0.0f) ? 1.0f : 0.0f;
```

## Rust FFI Changes

### New Parameters in persistent_ndt_launch

```rust
extern "C" {
    fn persistent_ndt_launch(
        // ... existing parameters ...

        // Debug output (Phase 19.4)
        debug_enabled: i32,
        debug_buffer: *mut f32,  // nullptr if disabled
    ) -> c_int;
}
```

### FullGpuPipelineV2 Changes

```rust
pub struct FullGpuPipelineV2 {
    // ... existing fields ...

    // Debug buffer (only allocated when enable_debug is true)
    persistent_debug_buffer: Option<Handle>,
}

impl FullGpuPipelineV2 {
    pub fn optimize(&mut self, ...) -> Result<FullGpuOptimizationResultV2> {
        // Allocate debug buffer only if enabled
        let debug_ptr = if self.config.enable_debug {
            let buffer_size = max_iterations as usize * 50 * std::mem::size_of::<f32>();
            self.persistent_debug_buffer = Some(self.client.empty(buffer_size));
            self.raw_ptr(self.persistent_debug_buffer.as_ref().unwrap())
        } else {
            0  // nullptr
        };

        // Launch kernel with debug parameters
        unsafe {
            cuda_ffi::persistent_ndt_launch_raw(
                // ... existing args ...
                if self.config.enable_debug { 1 } else { 0 },
                debug_ptr,
            )?;
        }

        // Parse debug data if enabled
        let iterations_debug = if self.config.enable_debug && iterations > 0 {
            Some(self.parse_debug_buffer(iterations as usize)?)
        } else {
            None
        };

        // ... rest of result construction ...
    }

    fn parse_debug_buffer(&self, num_iterations: usize) -> Result<Vec<IterationDebug>> {
        let buffer = self.persistent_debug_buffer.as_ref().unwrap();
        let bytes = self.client.read_one(buffer.clone());
        let floats = f32::from_bytes(&bytes);

        let mut result = Vec::with_capacity(num_iterations);
        for iter in 0..num_iterations {
            let base = iter * 50;
            let mut debug = IterationDebug::new(iter);

            // Parse fields from buffer
            debug.score = floats[base + 1] as f64;
            debug.pose = (0..6).map(|i| floats[base + 2 + i] as f64).collect();
            debug.gradient = (0..6).map(|i| floats[base + 8 + i] as f64).collect();

            // Expand Hessian from upper triangle
            debug.hessian = expand_upper_triangle(&floats[base + 14..base + 35]);

            debug.newton_step = (0..6).map(|i| floats[base + 35 + i] as f64).collect();
            debug.step_length = floats[base + 41] as f64;
            debug.num_correspondences = floats[base + 42] as usize;
            debug.direction_reversed = floats[base + 43] > 0.5;
            debug.pose_after = (0..6).map(|i| floats[base + 44 + i] as f64).collect();

            // Compute derived fields
            debug.newton_step_norm = debug.newton_step.iter()
                .map(|x| x * x).sum::<f64>().sqrt();
            debug.used_line_search = self.config.use_line_search;

            result.push(debug);
        }

        Ok(result)
    }
}
```

## Zero-Overhead Guarantee

When `debug_enabled == 0`:

1. **No buffer allocation** - `persistent_debug_buffer` is `None`
2. **No memory writes** - Single `if (debug_enabled)` check skips all debug writes
3. **No memory reads** - No download of debug buffer
4. **Minimal branch cost** - One predictable branch per iteration (always false)

The branch `if (debug_enabled && threadIdx.x == 0 && blockIdx.x == 0)` is:
- Only evaluated by thread 0 of block 0
- Highly predictable (always false when disabled)
- Short-circuits on first condition

## Implementation Order

1. Add `g_pose_before[6]` and `g_direction_reversed` to reduce_buffer state
2. Add debug parameters to kernel signature
3. Add debug write logic (guarded by debug_enabled)
4. Update FFI declarations
5. Add `persistent_debug_buffer: Option<Handle>` to pipeline
6. Add buffer allocation in optimize()
7. Add parse_debug_buffer() function
8. Update tests to verify debug data collection

## Testing

```rust
#[test]
fn test_debug_buffer_collection() {
    let config = PipelineV2Config {
        enable_debug: true,
        use_line_search: false,
        ..Default::default()
    };
    let mut pipeline = FullGpuPipelineV2::with_config(1000, 5000, config).unwrap();

    // ... set up data ...

    let result = pipeline.optimize(&initial_pose, 10, 0.01).unwrap();

    assert!(result.iterations_debug.is_some());
    let debug = result.iterations_debug.unwrap();
    assert_eq!(debug.len(), result.iterations as usize);

    // Verify first iteration has valid data
    let first = &debug[0];
    assert_eq!(first.iteration, 0);
    assert!(first.score.is_finite());
    assert_eq!(first.gradient.len(), 6);
    assert_eq!(first.hessian.len(), 36);
}
```

## Memory Impact

| Configuration | Buffer Size | Notes |
|--------------|-------------|-------|
| Disabled (default) | 0 bytes | No allocation |
| Enabled, 30 iterations | 6 KB | 50 × 30 × 4 bytes |
| Enabled, 100 iterations | 20 KB | 50 × 100 × 4 bytes |

The buffer is allocated once per pipeline and reused across alignments.
