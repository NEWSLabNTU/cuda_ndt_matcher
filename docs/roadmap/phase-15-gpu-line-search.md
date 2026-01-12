# Phase 15: True Full GPU Newton Pipeline with Line Search

## Overview

This phase implements a complete GPU Newton optimization pipeline with **zero CPU-GPU data transfers** during iterations. The design integrates:

1. **Zero-transfer iteration loop** - All Newton computation stays on GPU
2. **Batched More-Thuente line search** - K candidates evaluated in parallel
3. **GPU-resident state** - Pose, convergence, oscillation detection all on GPU

## Motivation

### Current Problems

The current `full_gpu_pipeline.rs` (Phase 14) has significant per-iteration transfers:

| Location      | Operation                                  | Transfer Size |
|---------------|--------------------------------------------|---------------|
| Lines 433-443 | Download J + PH, combine on CPU, re-upload | ~480 KB       |
| Line 534      | Download reduce_output                     | 172 bytes     |
| Line 550      | Download correspondences                   | ~3 KB         |
| Lines 343-345 | Upload transform matrix                    | 64 bytes      |
| Line 305      | Upload pose for sin/cos                    | 24 bytes      |

**Total per iteration**: ~490 KB

Additionally, no line search means:
- Fixed step size can overshoot (divergence)
- Fixed step size can undershoot (slow convergence)
- No Wolfe condition guarantees

### Goals

1. Reduce per-iteration transfer from ~490 KB to **4 bytes** (convergence flag only)
2. Add More-Thuente line search with **Strong Wolfe conditions**
3. Maintain mathematical correctness while maximizing GPU parallelism

## Architecture

### Complete GPU Iteration Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│              TRUE FULL GPU NEWTON PIPELINE WITH LINE SEARCH                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ONCE AT START (upload ~500 KB):                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │ Upload: source_points [N×3], voxel_data [V×12], initial_pose [6]           │  │
│  │ Upload: line search config, convergence epsilon                             │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  GPU ITERATION LOOP (CPU only launches kernels, no data transfer):               │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │ for iter in 0..max_iterations:                                              │  │
│  │                                                                              │  │
│  │   ╔═══════════════════════════════════════════════════════════════════════╗ │  │
│  │   ║ PHASE A: Compute Newton Direction δ = -H⁻¹g                           ║ │  │
│  │   ╠═══════════════════════════════════════════════════════════════════════╣ │  │
│  │   ║ 1. compute_sin_cos_kernel(pose_gpu → sin_cos_gpu)                     ║ │  │
│  │   ║ 2. compute_transform_kernel(sin_cos_gpu, pose_gpu → transform_gpu)    ║ │  │
│  │   ║ 3. compute_jacobians_kernel(sin_cos_gpu → jacobians_gpu)              ║ │  │
│  │   ║ 4. compute_point_hessians_kernel(sin_cos_gpu → phess_gpu)             ║ │  │
│  │   ║ 5. transform_points_kernel(transform_gpu → transformed_gpu)           ║ │  │
│  │   ║ 6. radius_search_kernel(→ neighbors_gpu)                              ║ │  │
│  │   ║ 7. compute_score_gradient_hessian_kernels(→ scores, grads, hess)      ║ │  │
│  │   ║ 8. CUB_segmented_reduce(→ score_gpu, gradient_gpu[6], H_gpu[36])      ║ │  │
│  │   ║ 9. cuSOLVER_solve_6x6(H_gpu, g_gpu → delta_gpu)                       ║ │  │
│  │   ╚═══════════════════════════════════════════════════════════════════════╝ │  │
│  │                                                                              │  │
│  │   ╔═══════════════════════════════════════════════════════════════════════╗ │  │
│  │   ║ PHASE B: Batched Line Search (find optimal step α)                    ║ │  │
│  │   ╠═══════════════════════════════════════════════════════════════════════╣ │  │
│  │   ║ 10. compute_dphi_0_kernel(gradient_gpu · delta_gpu → dphi_0_gpu)      ║ │  │
│  │   ║ 11. generate_candidates_kernel(→ candidates_gpu[K])                   ║ │  │
│  │   ║                                                                        ║ │  │
│  │   ║ 12. BATCH EVALUATE (parallel over K×N):                               ║ │  │
│  │   ║     batch_transform_kernel(pose, delta, candidates → K transforms)    ║ │  │
│  │   ║     batch_score_gradient_kernel(→ batch_scores[K×N], batch_grads)     ║ │  │
│  │   ║     batch_reduce(→ phi[K], dphi[K])                                   ║ │  │
│  │   ║                                                                        ║ │  │
│  │   ║ 13. more_thuente_kernel(phi_0, dphi_0, phi[K], dphi[K] → best_α)     ║ │  │
│  │   ╚═══════════════════════════════════════════════════════════════════════╝ │  │
│  │                                                                              │  │
│  │   ╔═══════════════════════════════════════════════════════════════════════╗ │  │
│  │   ║ PHASE C: Update State                                                  ║ │  │
│  │   ╠═══════════════════════════════════════════════════════════════════════╣ │  │
│  │   ║ 14. update_pose_kernel(pose_gpu += best_α × delta_gpu)                ║ │  │
│  │   ║ 15. check_convergence_kernel(best_α, delta_gpu → converged_flag)      ║ │  │
│  │   ╚═══════════════════════════════════════════════════════════════════════╝ │  │
│  │                                                                              │  │
│  │   // Only sync to check convergence flag (4 bytes)                          │  │
│  │   if converged_flag: break                                                  │  │
│  │                                                                              │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  ONCE AT END (download ~220 bytes):                                              │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │ Download: final_pose [48 bytes], score [8 bytes], H [288 bytes]            │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
                              GPU Memory Layout
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│  PERSISTENT BUFFERS (uploaded once):                                            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐              │
│  │ source_points    │  │ voxel_means      │  │ voxel_inv_covs   │              │
│  │ [N × 3]          │  │ [V × 3]          │  │ [V × 9]          │              │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘              │
│                                                                                  │
│  ITERATION STATE (GPU-resident):                                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐              │
│  │ pose_gpu [6]     │  │ delta_gpu [6]    │  │ best_alpha [1]   │              │
│  │ sin_cos [6]      │  │ transform [16]   │  │ converged [1]    │              │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘              │
│                                                                                  │
│  DERIVATIVE BUFFERS (reused each iteration):                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐              │
│  │ jacobians        │  │ point_hessians   │  │ transformed      │              │
│  │ [N × 18]         │  │ [N × 144]        │  │ [N × 3]          │              │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘              │
│                                                                                  │
│  REDUCTION BUFFERS:                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐              │
│  │ scores [N]       │  │ gradients [N×6]  │  │ hessians [N×36]  │              │
│  │     ↓            │  │     ↓            │  │     ↓            │              │
│  │ score_sum [1]    │  │ gradient [6]     │  │ H [36]           │              │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘              │
│                                                                                  │
│  LINE SEARCH BUFFERS:                                                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐              │
│  │ candidates [K]   │  │ batch_scores     │  │ batch_grads      │              │
│  │                  │  │ [K × N]          │  │ [K × N × 6]      │              │
│  │ phi_0, dphi_0    │  │     ↓            │  │     ↓            │              │
│  │ phi[K], dphi[K]  │  │ phi[K]           │  │ dphi[K]          │              │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Implementation

### Phase 15.1: GPU Transform Matrix Kernel

Move transform matrix computation from CPU to GPU.

```rust
/// Compute 4x4 transform matrix from sin/cos values on GPU.
#[cube(launch_unchecked)]
fn compute_transform_from_sincos_kernel<F: Float>(
    sin_cos: &Array<F>,      // [6]: sin_r, cos_r, sin_p, cos_p, sin_y, cos_y
    pose: &Array<F>,         // [6]: tx, ty, tz, roll, pitch, yaw
    transform: &mut Array<F>, // [16]: 4x4 row-major
) {
    if ABSOLUTE_POS != 0 { return; }

    let sr = sin_cos[0]; let cr = sin_cos[1];
    let sp = sin_cos[2]; let cp = sin_cos[3];
    let sy = sin_cos[4]; let cy = sin_cos[5];

    // Rz(yaw) * Ry(pitch) * Rx(roll)
    transform[0] = cy * cp;
    transform[1] = cy * sp * sr - sy * cr;
    transform[2] = cy * sp * cr + sy * sr;
    transform[3] = pose[0];  // tx

    transform[4] = sy * cp;
    transform[5] = sy * sp * sr + cy * cr;
    transform[6] = sy * sp * cr - cy * sr;
    transform[7] = pose[1];  // ty

    transform[8] = -sp;
    transform[9] = cp * sr;
    transform[10] = cp * cr;
    transform[11] = pose[2];  // tz

    transform[12] = F::new(0.0);
    transform[13] = F::new(0.0);
    transform[14] = F::new(0.0);
    transform[15] = F::new(1.0);
}
```

### Phase 15.2: Refactored Hessian Kernel (Separate Buffers)

Eliminate the CPU combine step by taking separate Jacobian and Point Hessian buffers.

```rust
/// Compute NDT Hessian with separate jacobian and point_hessian inputs.
/// Eliminates the need to combine buffers on CPU.
#[cube(launch_unchecked)]
fn compute_ndt_hessian_kernel_v2<F: Float>(
    source_points: &Array<F>,     // [N × 3]
    transform: &Array<F>,          // [16]
    jacobians: &Array<F>,          // [N × 18] - SEPARATE input
    point_hessians: &Array<F>,     // [N × 144] - SEPARATE input
    voxel_means: &Array<F>,        // [V × 3]
    voxel_inv_covs: &Array<F>,     // [V × 9]
    neighbor_indices: &Array<i32>, // [N × MAX_NEIGHBORS]
    neighbor_counts: &Array<u32>,  // [N]
    gauss_d1: F,
    gauss_d2: F,
    num_points: u32,
    hessians: &mut Array<F>,       // [N × 36] column-major
) {
    let point_idx = ABSOLUTE_POS as u32;
    if point_idx >= num_points { return; }

    // Read jacobians for this point (18 values)
    let j_offset = point_idx * 18;
    // ... read jacobians[j_offset..j_offset+18]

    // Read point_hessians for this point (144 values)
    let ph_offset = point_idx * 144;
    // ... read point_hessians[ph_offset..ph_offset+144]

    // Compute Hessian contribution using both buffers
    // ... (same math as before, just different input layout)
}
```

### Phase 15.3: GPU Newton Solve (Result Stays on GPU)

Modify cuSOLVER wrapper to keep the result on GPU.

```rust
impl GpuNewtonSolver {
    /// Solve 6×6 system entirely on GPU.
    /// Result stays in delta_gpu, no CPU download.
    ///
    /// Solves: H × δ = -g
    pub fn solve_inplace(
        &mut self,
        hessian_gpu_ptr: u64,   // Device pointer to 36 f64
        gradient_gpu_ptr: u64,  // Device pointer to 6 f64
        delta_gpu_ptr: u64,     // Device pointer to 6 f64 (output)
    ) -> Result<()> {
        // 1. Copy -gradient to delta (b = -g)
        // 2. Cholesky factorization: dpotrf(H)
        // 3. Solve: dpotrs(H, delta)
        // Result is in delta_gpu_ptr, no download needed
    }
}
```

### Phase 15.4: Directional Derivative Kernel

Compute φ'(0) = g · δ on GPU.

```rust
/// Compute dot product: result = a · b
#[cube(launch_unchecked)]
fn dot_product_6_kernel<F: Float>(
    a: &Array<F>,          // [6]
    b: &Array<F>,          // [6]
    result: &mut Array<F>, // [1]
) {
    if ABSOLUTE_POS != 0 { return; }

    result[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
              + a[3]*b[3] + a[4]*b[4] + a[5]*b[5];
}
```

### Phase 15.5: Candidate Generation Kernel

Generate K candidate step sizes for batched evaluation.

```rust
/// Generate K candidate step sizes for line search.
/// Strategy: Geometric progression centered on initial_step.
#[cube(launch_unchecked)]
fn generate_candidates_kernel<F: Float>(
    initial_step: F,
    step_min: F,
    step_max: F,
    num_candidates: u32,
    candidates: &mut Array<F>,  // [K]
) {
    let k = ABSOLUTE_POS as u32;
    if k >= num_candidates { return; }

    // Ratios: [1.0, 0.75, 0.5, 0.25, 0.125, 0.618, 0.382, 0.0625]
    let ratio = match k {
        0 => F::new(1.0),
        1 => F::new(0.75),
        2 => F::new(0.5),
        3 => F::new(0.25),
        4 => F::new(0.125),
        5 => F::new(0.618),  // Golden ratio
        6 => F::new(0.382),  // 1 - golden ratio
        7 => F::new(0.0625),
        _ => F::new(1.0),
    };

    let alpha = initial_step * ratio;
    candidates[k] = F::clamp(alpha, step_min, step_max);
}
```

### Phase 15.6: Batch Transform Kernel

Transform all points for all K candidates in parallel.

```rust
/// Transform points for K candidate step sizes.
/// Parallelism: K × N threads.
#[cube(launch_unchecked)]
fn batch_transform_kernel<F: Float>(
    source_points: &Array<F>,    // [N × 3]
    pose: &Array<F>,             // [6]
    delta: &Array<F>,            // [6]
    candidates: &Array<F>,       // [K]
    num_points: u32,
    num_candidates: u32,
    batch_transformed: &mut Array<F>,  // [K × N × 3]
) {
    let idx = ABSOLUTE_POS as u32;
    let k = idx / num_points;
    let i = idx % num_points;
    if k >= num_candidates { return; }

    let alpha = candidates[k];

    // Compute trial pose: pose + alpha * delta
    let trial_pose = [
        pose[0] + alpha * delta[0],
        pose[1] + alpha * delta[1],
        pose[2] + alpha * delta[2],
        pose[3] + alpha * delta[3],
        pose[4] + alpha * delta[4],
        pose[5] + alpha * delta[5],
    ];

    // Compute sin/cos for trial pose
    let sr = F::sin(trial_pose[3]); let cr = F::cos(trial_pose[3]);
    let sp = F::sin(trial_pose[4]); let cp = F::cos(trial_pose[4]);
    let sy = F::sin(trial_pose[5]); let cy = F::cos(trial_pose[5]);

    // Build rotation matrix
    let r00 = cy * cp;
    let r01 = cy * sp * sr - sy * cr;
    let r02 = cy * sp * cr + sy * sr;
    let r10 = sy * cp;
    let r11 = sy * sp * sr + cy * cr;
    let r12 = sy * sp * cr - cy * sr;
    let r20 = -sp;
    let r21 = cp * sr;
    let r22 = cp * cr;

    // Transform point
    let px = source_points[i * 3 + 0];
    let py = source_points[i * 3 + 1];
    let pz = source_points[i * 3 + 2];

    let out_idx = (k * num_points + i) * 3;
    batch_transformed[out_idx + 0] = r00*px + r01*py + r02*pz + trial_pose[0];
    batch_transformed[out_idx + 1] = r10*px + r11*py + r12*pz + trial_pose[1];
    batch_transformed[out_idx + 2] = r20*px + r21*py + r22*pz + trial_pose[2];
}
```

### Phase 15.7: Batch Score and Gradient Kernel

Compute score and gradient for all K candidates.

```rust
/// Compute NDT score and directional derivative for K candidates.
/// For line search, we only need score (φ) and directional derivative (φ').
#[cube(launch_unchecked)]
fn batch_score_gradient_kernel<F: Float>(
    batch_transformed: &Array<F>,  // [K × N × 3]
    delta: &Array<F>,              // [6] search direction
    jacobians: &Array<F>,          // [N × 18] (shared across candidates)
    voxel_means: &Array<F>,
    voxel_inv_covs: &Array<F>,
    neighbor_indices: &Array<i32>,
    neighbor_counts: &Array<u32>,
    gauss_d1: F,
    gauss_d2: F,
    num_points: u32,
    num_candidates: u32,
    batch_scores: &mut Array<F>,       // [K × N]
    batch_dir_derivs: &mut Array<F>,   // [K × N] directional derivatives
) {
    let idx = ABSOLUTE_POS as u32;
    let k = idx / num_points;
    let i = idx % num_points;
    if k >= num_candidates { return; }

    // Get transformed point for this candidate
    let tp_idx = (k * num_points + i) * 3;
    let tx = batch_transformed[tp_idx + 0];
    let ty = batch_transformed[tp_idx + 1];
    let tz = batch_transformed[tp_idx + 2];

    // Compute score and gradient contribution (reuse jacobians)
    let mut score = F::new(0.0);
    let mut grad = [F::new(0.0); 6];

    let num_neighbors = neighbor_counts[i];
    for n in 0..num_neighbors {
        let v_idx = neighbor_indices[i * MAX_NEIGHBORS + n];
        if v_idx < 0 { continue; }

        // ... compute score and gradient contribution from voxel v_idx
        // (same math as regular score/gradient kernels)
    }

    batch_scores[k * num_points + i] = score;

    // Directional derivative: grad · delta
    let dir_deriv = grad[0]*delta[0] + grad[1]*delta[1] + grad[2]*delta[2]
                  + grad[3]*delta[3] + grad[4]*delta[4] + grad[5]*delta[5];
    batch_dir_derivs[k * num_points + i] = dir_deriv;
}
```

### Phase 15.8: More-Thuente Logic Kernel

Run More-Thuente algorithm on GPU using cached candidate evaluations.

```rust
/// More-Thuente line search using pre-computed candidate evaluations.
/// Single-thread kernel that runs the sequential algorithm.
#[cube(launch_unchecked)]
fn more_thuente_kernel<F: Float>(
    phi_0: F,                    // Score at current pose
    dphi_0: F,                   // Directional derivative at current pose
    candidates: &Array<F>,       // [K] candidate step sizes
    cached_phi: &Array<F>,       // [K] scores for each candidate
    cached_dphi: &Array<F>,      // [K] directional derivatives
    num_candidates: u32,
    mu: F,                       // Sufficient decrease parameter (1e-4)
    nu: F,                       // Curvature parameter (0.9)
    max_iterations: u32,
    best_alpha: &mut Array<F>,   // [1] output
    ls_converged: &mut Array<F>, // [1] 1.0 if converged
) {
    if ABSOLUTE_POS != 0 { return; }

    // Initialize interval
    let mut a_l = F::new(0.0);
    let mut f_l = F::new(0.0);
    let mut g_l = dphi_0 - mu * dphi_0;

    let mut a_u = candidates[0];  // Start with largest candidate
    let mut f_u = F::new(0.0);
    let mut g_u = g_l;

    let mut a_t = candidates[0];
    let mut found_acceptable = false;

    for iter in 0..max_iterations {
        // Find closest cached candidate to current trial
        let (phi_t, dphi_t) = find_closest_cached(
            a_t, candidates, cached_phi, cached_dphi, num_candidates
        );

        // Compute ψ values (modified function)
        let psi_t = phi_t - phi_0 - mu * dphi_0 * a_t;
        let dpsi_t = dphi_t - mu * dphi_0;

        // Check Strong Wolfe conditions
        let armijo = phi_t <= phi_0 + mu * a_t * dphi_0;
        let curvature = F::abs(dphi_t) <= nu * F::abs(dphi_0);

        if armijo && curvature {
            best_alpha[0] = a_t;
            ls_converged[0] = F::new(1.0);
            return;
        }

        // Update interval based on More-Thuente cases
        // Case 1: ψ(a_t) > ψ(a_l)
        if psi_t > f_l {
            a_u = a_t;
            f_u = psi_t;
            g_u = dpsi_t;
        }
        // Case 2: ψ(a_t) ≤ ψ(a_l) and ψ'(a_t)(a_l - a_t) > 0
        else if dpsi_t * (a_l - a_t) > F::new(0.0) {
            a_l = a_t;
            f_l = psi_t;
            g_l = dpsi_t;
        }
        // Case 3: ψ(a_t) ≤ ψ(a_l) and ψ'(a_t)(a_l - a_t) ≤ 0
        else {
            a_u = a_l;
            f_u = f_l;
            g_u = g_l;
            a_l = a_t;
            f_l = psi_t;
            g_l = dpsi_t;
        }

        // Compute next trial via cubic interpolation
        a_t = cubic_interpolate(a_l, f_l, g_l, a_u, f_u, g_u);

        // Safeguard: ensure a_t is in interval
        let a_min = F::min(a_l, a_u);
        let a_max = F::max(a_l, a_u);
        a_t = F::clamp(a_t, a_min + F::new(0.001) * (a_max - a_min),
                            a_max - F::new(0.001) * (a_max - a_min));
    }

    // Max iterations reached - use best available
    // Find candidate with lowest score that satisfies Armijo
    let mut best_k = 0u32;
    let mut best_score = F::new(1e10);
    for k in 0..num_candidates {
        let phi_k = cached_phi[k];
        let a_k = candidates[k];
        if phi_k <= phi_0 + mu * a_k * dphi_0 && phi_k < best_score {
            best_score = phi_k;
            best_k = k;
        }
    }

    best_alpha[0] = candidates[best_k];
    ls_converged[0] = F::new(0.0);  // Did not fully converge
}

/// Find cached evaluation closest to target step size.
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

/// Cubic interpolation for next trial point.
fn cubic_interpolate<F: Float>(
    a_l: F, f_l: F, g_l: F,
    a_u: F, f_u: F, g_u: F,
) -> F {
    let d1 = g_l + g_u - F::new(3.0) * (f_l - f_u) / (a_l - a_u);
    let d2_sq = d1 * d1 - g_l * g_u;

    if d2_sq < F::new(0.0) {
        // Use bisection as fallback
        return (a_l + a_u) / F::new(2.0);
    }

    let d2 = F::sqrt(d2_sq);
    let a_c = a_u - (a_u - a_l) * (g_u + d2 - d1) / (g_u - g_l + F::new(2.0) * d2);

    a_c
}
```

### Phase 15.9: Pose Update Kernel

Update pose with optimal step size.

```rust
/// Update pose: pose += alpha * delta
#[cube(launch_unchecked)]
fn update_pose_kernel<F: Float>(
    pose: &mut Array<F>,   // [6] modified in place
    delta: &Array<F>,      // [6]
    alpha: &Array<F>,      // [1] step size
) {
    let i = ABSOLUTE_POS as u32;
    if i >= 6 { return; }

    pose[i] = pose[i] + alpha[0] * delta[i];
}
```

### Phase 15.10: Convergence Check Kernel

Check convergence based on step size.

```rust
/// Check convergence: ||alpha * delta|| < epsilon
#[cube(launch_unchecked)]
fn check_convergence_kernel<F: Float>(
    delta: &Array<F>,          // [6]
    alpha: &Array<F>,          // [1]
    epsilon_sq: F,
    converged: &mut Array<u32>, // [1]
) {
    if ABSOLUTE_POS != 0 { return; }

    let a = alpha[0];
    let step_sq = (a * delta[0]) * (a * delta[0])
                + (a * delta[1]) * (a * delta[1])
                + (a * delta[2]) * (a * delta[2])
                + (a * delta[3]) * (a * delta[3])
                + (a * delta[4]) * (a * delta[4])
                + (a * delta[5]) * (a * delta[5]);

    converged[0] = if step_sq < epsilon_sq { 1u32 } else { 0u32 };
}
```

### Phase 15.11: Integrated Pipeline

```rust
/// Full GPU optimization pipeline with integrated line search.
pub struct FullGpuPipelineV2 {
    client: CudaClient,

    // Persistent data
    source_points: Handle,    // [N × 3]
    voxel_means: Handle,      // [V × 3]
    voxel_inv_covs: Handle,   // [V × 9]

    // Iteration state (GPU-resident)
    pose_gpu: Handle,         // [6]
    delta_gpu: Handle,        // [6]
    sin_cos: Handle,          // [6]
    transform: Handle,        // [16]

    // Derivative buffers
    jacobians: Handle,        // [N × 18]
    point_hessians: Handle,   // [N × 144]
    transformed: Handle,      // [N × 3]
    neighbor_indices: Handle, // [N × MAX_NEIGHBORS]
    neighbor_counts: Handle,  // [N]

    // Reduction buffers
    scores: Handle,           // [N]
    gradients: Handle,        // [N × 6]
    hessians: Handle,         // [N × 36]
    reduce_output: Handle,    // [43]: score + grad[6] + H[36]

    // Line search buffers
    candidates: Handle,        // [K]
    batch_transformed: Handle, // [K × N × 3]
    batch_scores: Handle,      // [K × N]
    batch_dir_derivs: Handle,  // [K × N]
    phi_cache: Handle,         // [K]
    dphi_cache: Handle,        // [K]
    phi_0: Handle,             // [1]
    dphi_0: Handle,            // [1]
    best_alpha: Handle,        // [1]

    // Convergence flag
    converged_flag: Handle,    // [1]

    // Solvers
    newton_solver: GpuNewtonSolver,

    // Configuration
    num_candidates: u32,       // K = 8
    gauss_d1: f32,
    gauss_d2: f32,
    search_radius_sq: f32,
}

impl FullGpuPipelineV2 {
    pub fn optimize(
        &mut self,
        initial_pose: &[f64; 6],
        max_iterations: u32,
        transformation_epsilon: f64,
    ) -> Result<FullGpuOptimizationResult> {
        // Upload initial pose ONCE
        let pose_f32: [f32; 6] = initial_pose.map(|x| x as f32);
        self.pose_gpu = self.client.create(f32::as_bytes(&pose_f32));

        let epsilon_sq = (transformation_epsilon * transformation_epsilon) as f32;
        let mut iterations = 0u32;

        for iter in 0..max_iterations {
            iterations = iter + 1;

            // ═══════════════════════════════════════════════════════════
            // PHASE A: Compute Newton direction δ = -H⁻¹g
            // ═══════════════════════════════════════════════════════════

            // 1. Compute sin/cos from pose
            self.launch_sin_cos_kernel();

            // 2. Compute transform matrix
            self.launch_transform_kernel();

            // 3. Compute Jacobians
            self.launch_jacobians_kernel();

            // 4. Compute Point Hessians
            self.launch_point_hessians_kernel();

            // 5. Transform points
            self.launch_transform_points_kernel();

            // 6. Radius search
            self.launch_radius_search_kernel();

            // 7. Compute scores, gradients, hessians
            self.launch_score_kernel();
            self.launch_gradient_kernel();
            self.launch_hessian_kernel_v2();  // Uses separate J, PH buffers

            // 8. CUB reduction
            self.launch_cub_reduction();

            // 9. Newton solve (result stays on GPU)
            self.newton_solver.solve_inplace(
                self.raw_ptr(&self.hessian_gpu),
                self.raw_ptr(&self.gradient_gpu),
                self.raw_ptr(&self.delta_gpu),
            )?;

            // ═══════════════════════════════════════════════════════════
            // PHASE B: Batched line search
            // ═══════════════════════════════════════════════════════════

            // 10. Compute directional derivative: φ'(0) = g · δ
            self.launch_dot_product_kernel();

            // 11. Generate K candidates
            self.launch_generate_candidates_kernel();

            // 12. Batch evaluate all candidates
            self.launch_batch_transform_kernel();
            self.launch_batch_score_gradient_kernel();
            self.launch_batch_reduce();

            // 13. More-Thuente on GPU
            self.launch_more_thuente_kernel();

            // ═══════════════════════════════════════════════════════════
            // PHASE C: Update state
            // ═══════════════════════════════════════════════════════════

            // 14. Update pose: pose += best_α × δ
            self.launch_update_pose_kernel();

            // 15. Check convergence
            self.launch_convergence_kernel(epsilon_sq);

            // Download ONLY convergence flag (4 bytes)
            let flag_bytes = self.client.read_one(self.converged_flag.clone());
            if u32::from_bytes(&flag_bytes)[0] != 0 {
                break;
            }
        }

        // Download final results ONCE
        self.download_final_results(iterations)
    }
}
```

## Transfer Analysis

### Comparison

| Phase | Per-Iteration Transfer | 30 Iterations | Notes |
|-------|------------------------|---------------|-------|
| Phase 14 (current) | ~490 KB | ~15 MB | J/PH combine roundtrip |
| Phase 15 (new) | 4 bytes | 120 bytes | Convergence flag only |
| **Improvement** | **122,500×** | **125,000×** | |

### Memory Overhead

| Buffer | Size (N=756, K=8) | Notes |
|--------|-------------------|-------|
| batch_transformed | K×N×3×4 = 72.6 KB | Line search transforms |
| batch_scores | K×N×4 = 24.2 KB | Line search scores |
| batch_dir_derivs | K×N×4 = 24.2 KB | Line search derivatives |
| phi_cache, dphi_cache | K×4×2 = 64 B | Reduced values |
| candidates | K×4 = 32 B | Step sizes |
| **Total overhead** | ~121 KB | Acceptable |

## Testing

### Unit Tests

1. `test_transform_kernel` - GPU transform matches CPU
2. `test_hessian_kernel_v2` - Separate buffers produce same result
3. `test_dot_product_kernel` - Verify φ'(0) computation
4. `test_candidate_generation` - Verify K candidates generated
5. `test_batch_transform` - K transforms match K sequential
6. `test_batch_score_gradient` - Batch results match single
7. `test_more_thuente_kernel` - Wolfe conditions satisfied
8. `test_update_pose_kernel` - pose += α×δ correct
9. `test_convergence_kernel` - Threshold check correct

### Integration Tests

1. `test_zero_transfer_vs_current` - Results match Phase 14
2. `test_line_search_improves_convergence` - Better than fixed step
3. `test_wolfe_conditions_satisfied` - Line search output valid
4. `test_full_alignment_accuracy` - Match Autoware results

### Performance Tests

1. Measure per-iteration latency
2. Profile GPU kernel utilization
3. Verify transfer reduction
4. Compare convergence rate with/without line search

## Implementation Order

1. **15.1**: GPU transform kernel
2. **15.2**: Refactored Hessian kernel (separate buffers)
3. **15.3**: GPU Newton solve (result on GPU)
4. **15.4**: Directional derivative kernel
5. **15.5**: Candidate generation kernel
6. **15.6**: Batch transform kernel
7. **15.7**: Batch score/gradient kernel
8. **15.8**: More-Thuente logic kernel
9. **15.9**: Pose update kernel
10. **15.10**: Convergence check kernel
11. **15.11**: Integrated pipeline

## Status

- [ ] 15.1: GPU transform kernel
- [ ] 15.2: Hessian kernel v2 (separate buffers)
- [ ] 15.3: GPU Newton solve (result on GPU)
- [ ] 15.4: Directional derivative kernel
- [ ] 15.5: Candidate generation kernel
- [ ] 15.6: Batch transform kernel
- [ ] 15.7: Batch score/gradient kernel
- [ ] 15.8: More-Thuente logic kernel
- [ ] 15.9: Pose update kernel
- [ ] 15.10: Convergence check kernel
- [ ] 15.11: Integrated pipeline
- [ ] Unit tests
- [ ] Integration tests
- [ ] Performance benchmarks
