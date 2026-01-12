# Phase 15: GPU-Accelerated More-Thuente Line Search

## Overview

This phase adds GPU-accelerated line search to the full GPU Newton pipeline, maintaining mathematical correctness of the More-Thuente algorithm while maximizing parallelism.

## Motivation

The current full GPU pipeline (Phase 14) uses a fixed step size, which can:
- Overshoot the optimum (divergence)
- Undershoot (slow convergence)
- Miss the optimal step that satisfies Wolfe conditions

More-Thuente line search finds the optimal step size but is inherently sequential. This phase implements a hybrid approach that:
- Maintains exact More-Thuente algorithm (mathematical correctness)
- Batches candidate evaluations for GPU parallelism
- Keeps all computation on GPU (zero per-iteration transfers)

## Background: More-Thuente Algorithm

### Wolfe Conditions

The algorithm finds step size α satisfying:

```
1. Sufficient Decrease (Armijo):
   φ(α) ≤ φ(0) + μ · α · φ'(0)

2. Curvature Condition:
   |φ'(α)| ≤ ν · |φ'(0)|
```

Where:
- φ(α) = score at pose + α × direction
- φ'(α) = directional derivative = gradient(α) · direction
- μ = 1e-4 (sufficient decrease parameter)
- ν = 0.9 (curvature parameter)

### Algorithm Structure

```
Initialize: interval [a_l, a_u], trial α

For each iteration:
  1. Evaluate φ(α), φ'(α)           ← EXPENSIVE (full derivative computation)
  2. Check Wolfe conditions         ← TRIVIAL (2 comparisons)
  3. Update interval (Cases 1-4)    ← TRIVIAL (6 assignments)
  4. Interpolate next trial α       ← TRIVIAL (cubic/quadratic formula)
```

Key insight: Step 1 is 99.9% of the work and is fully parallelizable.

## Design: Batched Speculative Evaluation

### Approach

Instead of evaluating one candidate at a time, we:
1. Generate K candidate step sizes that likely cover the optimal α
2. Evaluate ALL K candidates in parallel on GPU
3. Run More-Thuente iterations using cached results
4. Only generate new batch if needed (rare, <5% of cases)

### Candidate Generation Strategy

```
Initial batch (K=8):
  α₀ = initial_step (e.g., 0.1 from config)

  Candidates: [α₀, α₀×0.75, α₀×0.5, α₀×0.25, α₀×0.125,
               α₀×0.618, α₀×0.382, α₀×0.0625]

  Rationale:
  - Geometric series covers exponential range
  - Golden ratio points (0.618, 0.382) match interpolation behavior
  - 8 candidates typically sufficient for convergence
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU Line Search Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. GENERATE CANDIDATES (GPU, parallel over K)                  │
│     ┌─────────────────────────────────────────┐                 │
│     │ candidates[k] = α₀ × ratio[k]           │                 │
│     │ Output: candidates[K]                    │                 │
│     └─────────────────────────────────────────┘                 │
│                                                                  │
│  2. BATCH TRANSFORM (GPU, parallel over K×N)                    │
│     ┌─────────────────────────────────────────┐                 │
│     │ For each (k, i):                         │                 │
│     │   trial_pose = pose + candidates[k] × δ  │                 │
│     │   transformed[k,i] = transform(point[i]) │                 │
│     │ Output: transformed[K×N×3]               │                 │
│     └─────────────────────────────────────────┘                 │
│                                                                  │
│  3. BATCH DERIVATIVES (GPU, parallel over K×N×V)                │
│     ┌─────────────────────────────────────────┐                 │
│     │ For each (k, i, v):                      │                 │
│     │   score_contrib = ndt_score(...)         │                 │
│     │   grad_contrib = ndt_gradient(...)       │                 │
│     │ Output: scores[K×N], gradients[K×N×6]    │                 │
│     └─────────────────────────────────────────┘                 │
│                                                                  │
│  4. BATCH REDUCE (GPU, K parallel reductions)                   │
│     ┌─────────────────────────────────────────┐                 │
│     │ For each k:                              │                 │
│     │   φ(α_k) = Σᵢ scores[k,i]               │                 │
│     │   g_k = Σᵢ gradients[k,i]               │                 │
│     │   φ'(α_k) = g_k · direction              │                 │
│     │ Output: cached_phi[K], cached_dphi[K]    │                 │
│     └─────────────────────────────────────────┘                 │
│                                                                  │
│  5. MORE-THUENTE ITERATIONS (GPU, single thread)                │
│     ┌─────────────────────────────────────────┐                 │
│     │ For iter in 0..max_iters:                │                 │
│     │   α = current trial                      │                 │
│     │   (φ, φ') = lookup(α, cached_*)         │                 │
│     │   if wolfe_satisfied: return α           │                 │
│     │   update_interval(...)                   │                 │
│     │   α = interpolate_next(...)              │                 │
│     │ Output: best_α, converged                │                 │
│     └─────────────────────────────────────────┘                 │
│                                                                  │
│  6. UPDATE POSE (GPU, parallel over 6)                          │
│     ┌─────────────────────────────────────────┐                 │
│     │ pose[i] += best_α × direction[i]         │                 │
│     └─────────────────────────────────────────┘                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation

### New Files

```
src/ndt_cuda/src/optimization/
├── gpu_line_search.rs      # GPU line search implementation
└── mod.rs                  # Add module export
```

### Data Structures

```rust
/// Configuration for GPU line search
#[derive(Debug, Clone)]
pub struct GpuLineSearchConfig {
    /// Maximum step length (default: 0.1)
    pub step_max: f32,
    /// Minimum step length (default: 1e-9)
    pub step_min: f32,
    /// Sufficient decrease parameter μ (default: 1e-4)
    pub mu: f32,
    /// Curvature parameter ν (default: 0.9)
    pub nu: f32,
    /// Maximum line search iterations (default: 10)
    pub max_iterations: u32,
    /// Number of candidates per batch (default: 8)
    pub num_candidates: u32,
}

/// GPU buffers for line search
pub struct GpuLineSearchPipeline {
    // Configuration
    config: GpuLineSearchConfig,

    // Candidate buffers
    candidates: Handle,           // [K] step sizes

    // Batch evaluation buffers
    candidate_transforms: Handle, // [K × 16] transform matrices
    transformed_points: Handle,   // [K × N × 3]
    batch_scores: Handle,         // [K × N]
    batch_gradients: Handle,      // [K × N × 6]

    // Reduction output
    cached_phi: Handle,           // [K] scores
    cached_dphi: Handle,          // [K] directional derivatives

    // More-Thuente state
    interval: Handle,             // [6] a_l, f_l, g_l, a_u, f_u, g_u

    // Result
    result: Handle,               // [3] best_α, converged, iterations

    // CUB reduction workspace
    reduce_temp: Handle,
    reduce_offsets: Handle,
}
```

### GPU Kernels

#### 1. Generate Candidates

```rust
#[cube(launch_unchecked)]
fn generate_candidates_kernel<F: Float>(
    initial_step: F,
    step_min: F,
    step_max: F,
    ratios: &Array<F>,      // Pre-defined: [1.0, 0.75, 0.5, 0.25, 0.125, 0.618, 0.382, 0.0625]
    num_candidates: u32,
    candidates: &mut Array<F>,
) {
    let k = ABSOLUTE_POS as u32;
    if k >= num_candidates { return; }

    let alpha = initial_step * ratios[k];
    candidates[k] = alpha.clamp(step_min, step_max);
}
```

#### 2. Batch Transform

```rust
#[cube(launch_unchecked)]
fn batch_transform_kernel<F: Float>(
    source_points: &Array<F>,    // [N × 3]
    pose: &Array<F>,             // [6]
    direction: &Array<F>,        // [6]
    candidates: &Array<F>,       // [K]
    num_points: u32,
    num_candidates: u32,
    transformed: &mut Array<F>,  // [K × N × 3]
) {
    let idx = ABSOLUTE_POS as u32;
    let k = idx / num_points;
    let i = idx % num_points;
    if k >= num_candidates { return; }

    let alpha = candidates[k];

    // Compute trial pose: pose + alpha * direction
    let trial_pose = [
        pose[0] + alpha * direction[0],
        pose[1] + alpha * direction[1],
        pose[2] + alpha * direction[2],
        pose[3] + alpha * direction[3],
        pose[4] + alpha * direction[4],
        pose[5] + alpha * direction[5],
    ];

    // Compute transform matrix from trial_pose
    // (sin/cos computation + rotation matrix)
    let transform = pose_to_transform(&trial_pose);

    // Transform point
    let px = source_points[i * 3 + 0];
    let py = source_points[i * 3 + 1];
    let pz = source_points[i * 3 + 2];

    let out_idx = (k * num_points + i) * 3;
    transformed[out_idx + 0] = transform[0]*px + transform[1]*py + transform[2]*pz + transform[3];
    transformed[out_idx + 1] = transform[4]*px + transform[5]*py + transform[6]*pz + transform[7];
    transformed[out_idx + 2] = transform[8]*px + transform[9]*py + transform[10]*pz + transform[11];
}
```

#### 3. More-Thuente Logic Kernel

```rust
#[cube(launch_unchecked)]
fn more_thuente_kernel<F: Float>(
    phi_0: F,
    dphi_0: F,
    candidates: &Array<F>,
    cached_phi: &Array<F>,
    cached_dphi: &Array<F>,
    num_candidates: u32,
    mu: F,
    nu: F,
    max_iterations: u32,
    result: &mut Array<F>,  // [best_α, converged, iterations]
) {
    // Single thread executes More-Thuente logic
    if ABSOLUTE_POS != 0 { return; }

    // Initialize interval [a_l, a_u]
    let mut a_l = F::new(0.0);
    let mut f_l = F::new(0.0);  // psi(0) = 0
    let mut g_l = dphi_0 - mu * dphi_0;  // psi'(0)

    let mut a_u = F::new(0.0);
    let mut f_u = F::new(0.0);
    let mut g_u = g_l;

    let mut a_t = candidates[0];  // Start with first candidate
    let mut open_interval = true;

    for iter in 0..max_iterations {
        // Find closest cached value to a_t
        let (phi_t, dphi_t) = find_closest_cached(
            a_t, candidates, cached_phi, cached_dphi, num_candidates
        );

        // Compute psi values
        let psi_t = phi_t - phi_0 - mu * dphi_0 * a_t;
        let dpsi_t = dphi_t - mu * dphi_0;

        // Check Wolfe conditions
        let sufficient_decrease = psi_t <= F::new(0.0);
        let curvature = F::abs(dphi_t) <= nu * F::abs(dphi_0);

        if sufficient_decrease && curvature {
            result[0] = a_t;
            result[1] = F::new(1.0);  // converged
            result[2] = F::cast_from(iter);
            return;
        }

        // Update interval (Cases 1-4 from More-Thuente)
        // ... (interval update logic)

        // Select next trial via interpolation
        a_t = trial_value_selection(a_l, f_l, g_l, a_u, f_u, g_u, a_t,
                                    if open_interval { psi_t } else { phi_t },
                                    if open_interval { dpsi_t } else { dphi_t });
    }

    // Max iterations reached
    result[0] = a_t;
    result[1] = F::new(0.0);  // not converged
    result[2] = F::cast_from(max_iterations);
}

/// Find closest cached candidate to target step size
fn find_closest_cached<F: Float>(
    target: F,
    candidates: &Array<F>,
    cached_phi: &Array<F>,
    cached_dphi: &Array<F>,
    num_candidates: u32,
) -> (F, F) {
    let mut best_k = 0u32;
    let mut best_dist = F::new(1e10);

    for k in 0..num_candidates {
        let dist = F::abs(candidates[k] - target);
        if dist < best_dist {
            best_dist = dist;
            best_k = k;
        }
    }

    (cached_phi[best_k], cached_dphi[best_k])
}
```

### Integration with FullGpuPipeline

```rust
impl FullGpuPipeline {
    /// Run Newton iteration with GPU line search
    pub fn iterate_with_line_search(&mut self) -> Result<ConvergenceStatus> {
        // 1. Compute score, gradient, Hessian at current pose
        self.compute_sin_cos();
        self.compute_jacobians();
        self.compute_point_hessians();
        self.transform_points();
        self.radius_search();
        self.compute_scores();
        self.compute_gradients();
        self.compute_hessians();
        self.reduce_all();

        // 2. Solve Newton direction: δ = -H⁻¹g
        self.newton_solver.solve(&self.hessian, &self.gradient, &self.direction)?;

        // 3. Compute initial directional derivative: φ'(0) = g · δ
        let dphi_0 = self.compute_directional_derivative();

        // 4. GPU Line Search
        self.line_search.generate_candidates(self.config.initial_step);
        self.line_search.batch_evaluate(
            &self.source_points,
            &self.pose,
            &self.direction,
            &self.voxel_data,
        );
        self.line_search.run_more_thuente(
            self.current_score,
            dphi_0,
        );

        // 5. Update pose: pose += best_α × direction
        self.apply_step(&self.line_search.best_alpha, &self.direction);

        // 6. Check convergence
        self.check_convergence()
    }
}
```

## Complexity Analysis

### Compute

| Operation | Work | Parallelism |
|-----------|------|-------------|
| Generate K candidates | K | K-way parallel |
| Transform K×N points | K×N×12 ops | K×N-way parallel |
| Radius search | K×N×V | K×N-way parallel |
| Score/gradient | K×N×V×10 ops | K×N-way parallel |
| Reduce | K×N → K | K parallel reductions |
| More-Thuente logic | ~250 ops | Sequential (trivial) |

**Total**: O(K × N × V) with full parallelism, where K=8, N=756 (typical), V=12000 (typical)

### Memory

| Buffer | Size | Notes |
|--------|------|-------|
| candidates | K×4 bytes | 32 bytes |
| transformed_points | K×N×3×4 bytes | ~73 KB (K=8, N=756) |
| batch_scores | K×N×4 bytes | ~24 KB |
| batch_gradients | K×N×6×4 bytes | ~145 KB |
| cached results | K×2×4 bytes | 64 bytes |

**Total overhead**: ~250 KB (acceptable)

### Transfers

| Direction | Data | Size |
|-----------|------|------|
| Per Newton iteration | None | 0 bytes |
| After convergence | Final pose | 48 bytes |

## Comparison

| Aspect | Fixed Step | CPU More-Thuente | GPU More-Thuente |
|--------|------------|------------------|------------------|
| Mathematical correctness | ❌ | ✅ | ✅ |
| Wolfe conditions | ❌ | ✅ | ✅ |
| Per-iteration transfers | 0 | ~100 bytes × iters | 0 |
| Compute overhead | 1× | K× (sequential) | K× (parallel) |
| Convergence quality | Poor | Optimal | Optimal |

## Future Optimizations

### Adaptive Candidate Count

Start with fewer candidates (K=4), increase if cache miss:
```rust
if !found_in_cache(interpolated_alpha) {
    generate_refined_candidates(interpolated_alpha);
    batch_evaluate_new_candidates();
}
```

### Shared Memory Optimization

Cache frequently accessed values in shared memory:
- Candidate step sizes
- Voxel data for hot voxels

### Convergence Prediction

Track typical convergence patterns, pre-warm cache:
```rust
if historical_convergence_at_alpha_range(0.05..0.08) {
    focus_candidates_on_range(0.05, 0.08);
}
```

## Testing

### Unit Tests

1. `test_candidate_generation` - Verify candidate step sizes
2. `test_batch_transform` - Verify K transforms match K sequential transforms
3. `test_batch_score` - Verify batch scores match single evaluation
4. `test_wolfe_conditions` - Verify condition checking
5. `test_interval_update` - Verify Cases 1-4
6. `test_interpolation` - Verify cubic/quadratic formulas
7. `test_full_line_search` - End-to-end on known functions

### Integration Tests

1. `test_line_search_improves_convergence` - Compare with fixed step
2. `test_line_search_vs_cpu` - Verify GPU matches CPU More-Thuente
3. `test_alignment_with_line_search` - Full NDT alignment

## Status

- [ ] Design document (this file)
- [ ] Candidate generation kernel
- [ ] Batch transform kernel
- [ ] Batch score/gradient kernels
- [ ] Batch reduction
- [ ] More-Thuente logic kernel
- [ ] Integration with FullGpuPipeline
- [ ] Unit tests
- [ ] Integration tests
- [ ] Performance benchmarking
