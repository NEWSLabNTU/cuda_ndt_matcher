// 6x6 Cholesky solver in registers for GPU
//
// This implements Cholesky factorization and solve for 6x6 symmetric positive
// definite matrices entirely in registers. Used for Newton step computation
// in the persistent NDT kernel.

#ifndef CHOLESKY_6X6_CUH
#define CHOLESKY_6X6_CUH

#include <cuda_runtime.h>
#include <cmath>

// ============================================================================
// 6x6 Cholesky decomposition and solve (double precision)
// ============================================================================

/// Solve H * x = -g using Cholesky decomposition.
///
/// The Hessian H must be symmetric positive definite.
/// On success, g is overwritten with the solution x.
/// H is destroyed during factorization.
///
/// @param H     6x6 Hessian matrix, row-major [36 doubles]. Destroyed.
/// @param g     6-element gradient vector. Overwritten with solution.
/// @param success  Output: true if solve succeeded, false if H is not SPD.
__device__ __forceinline__ void cholesky_solve_6x6_f64(
    double* H,        // [36] row-major, destroyed
    double* g,        // [6] gradient in, solution out
    bool* success
) {
    // Cholesky factorization: H = L * L^T
    // We store L in-place in the lower triangle of H

    for (int j = 0; j < 6; j++) {
        // Compute diagonal element L[j][j]
        double sum = H[j * 6 + j];
        for (int k = 0; k < j; k++) {
            double Ljk = H[j * 6 + k];
            sum -= Ljk * Ljk;
        }

        if (sum <= 1e-10) {
            // Matrix is not positive definite (or nearly singular)
            *success = false;
            return;
        }

        double Ljj = sqrt(sum);
        H[j * 6 + j] = Ljj;

        // Compute off-diagonal elements L[i][j] for i > j
        for (int i = j + 1; i < 6; i++) {
            sum = H[i * 6 + j];
            for (int k = 0; k < j; k++) {
                sum -= H[i * 6 + k] * H[j * 6 + k];
            }
            H[i * 6 + j] = sum / Ljj;
        }
    }

    // Forward substitution: L * y = -g
    for (int i = 0; i < 6; i++) {
        double sum = -g[i];  // Note: negative for Newton step
        for (int j = 0; j < i; j++) {
            sum -= H[i * 6 + j] * g[j];
        }
        g[i] = sum / H[i * 6 + i];
    }

    // Backward substitution: L^T * x = y
    for (int i = 5; i >= 0; i--) {
        double sum = g[i];
        for (int j = i + 1; j < 6; j++) {
            sum -= H[j * 6 + i] * g[j];  // L^T[i][j] = L[j][i]
        }
        g[i] = sum / H[i * 6 + i];
    }

    *success = true;
}

// ============================================================================
// Cholesky with Levenberg-Marquardt regularization
// ============================================================================

/// Solve H * x = -g with diagonal regularization for indefinite matrices.
///
/// If the matrix is not positive definite, adds increasing regularization
/// to the diagonal until Cholesky succeeds. This handles the case where
/// NDT Hessian is indefinite (e.g., far from optimum or at saddle points).
///
/// @param H_orig  6x6 Hessian matrix, row-major [36 doubles]. Not modified.
/// @param g_orig  6-element gradient vector [6 doubles]. Not modified.
/// @param x_out   6-element solution output.
/// @param success Output: true if solve succeeded.
__device__ __forceinline__ void cholesky_solve_regularized_6x6_f64(
    const double* H_orig,  // [36] row-major, not modified
    const double* g_orig,  // [6] gradient
    double* x_out,         // [6] solution output
    bool* success
) {
    // Working copies
    double H[36];
    double g[6];

    // Initial regularization based on Hessian magnitude
    double max_diag = 0.0;
    for (int i = 0; i < 6; i++) {
        double d = fabs(H_orig[i * 6 + i]);
        if (d > max_diag) max_diag = d;
    }

    // Try with increasing regularization: 0, 1e-6*max, 1e-4*max, 1e-2*max, 1*max
    double reg_factors[5] = {0.0, 1e-6, 1e-4, 1e-2, 1.0};

    for (int attempt = 0; attempt < 5; attempt++) {
        double reg = reg_factors[attempt] * max_diag;
        if (reg < 1e-6 && attempt > 0) reg = 1e-6;  // Minimum regularization

        // Copy with regularization
        for (int i = 0; i < 36; i++) H[i] = H_orig[i];
        for (int i = 0; i < 6; i++) {
            H[i * 6 + i] += reg;
            g[i] = g_orig[i];
        }

        // Try Cholesky
        bool chol_success = true;
        for (int j = 0; j < 6; j++) {
            double sum = H[j * 6 + j];
            for (int k = 0; k < j; k++) {
                double Ljk = H[j * 6 + k];
                sum -= Ljk * Ljk;
            }

            if (sum <= 1e-10) {
                chol_success = false;
                break;
            }

            double Ljj = sqrt(sum);
            H[j * 6 + j] = Ljj;

            for (int i = j + 1; i < 6; i++) {
                sum = H[i * 6 + j];
                for (int k = 0; k < j; k++) {
                    sum -= H[i * 6 + k] * H[j * 6 + k];
                }
                H[i * 6 + j] = sum / Ljj;
            }
        }

        if (!chol_success) continue;

        // Forward substitution: L * y = -g
        for (int i = 0; i < 6; i++) {
            double sum = -g[i];
            for (int j = 0; j < i; j++) {
                sum -= H[i * 6 + j] * g[j];
            }
            g[i] = sum / H[i * 6 + i];
        }

        // Backward substitution: L^T * x = y
        for (int i = 5; i >= 0; i--) {
            double sum = g[i];
            for (int j = i + 1; j < 6; j++) {
                sum -= H[j * 6 + i] * g[j];
            }
            g[i] = sum / H[i * 6 + i];
        }

        // Copy solution
        for (int i = 0; i < 6; i++) x_out[i] = g[i];
        *success = true;
        return;
    }

    // All attempts failed
    *success = false;
}

// ============================================================================
// Expand upper triangle to full symmetric matrix
// ============================================================================

/// Expand 21-element upper triangle to 36-element full symmetric matrix.
///
/// Upper triangle order: (0,0), (0,1), (0,2), (0,3), (0,4), (0,5),
///                              (1,1), (1,2), (1,3), (1,4), (1,5),
///                                     (2,2), (2,3), (2,4), (2,5),
///                                            (3,3), (3,4), (3,5),
///                                                   (4,4), (4,5),
///                                                          (5,5)
__device__ __forceinline__ void expand_upper_triangle_f32_to_f64(
    const float* upper,  // [21] upper triangle
    double* full         // [36] output, row-major
) {
    int idx = 0;
    for (int i = 0; i < 6; i++) {
        for (int j = i; j < 6; j++) {
            double val = (double)upper[idx++];
            full[i * 6 + j] = val;
            full[j * 6 + i] = val;  // Symmetric
        }
    }
}

/// Expand 21-element upper triangle to 36-element full symmetric matrix (f32).
__device__ __forceinline__ void expand_upper_triangle_f32(
    const float* upper,  // [21] upper triangle
    float* full          // [36] output, row-major
) {
    int idx = 0;
    for (int i = 0; i < 6; i++) {
        for (int j = i; j < 6; j++) {
            float val = upper[idx++];
            full[i * 6 + j] = val;
            full[j * 6 + i] = val;  // Symmetric
        }
    }
}

#endif // CHOLESKY_6X6_CUH
