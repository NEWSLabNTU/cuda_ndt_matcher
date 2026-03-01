//! Newton-based optimization for NDT scan matching.
//!
//! This module implements the optimization loop for NDT:
//! 1. Transform source points using current pose
//! 2. Compute derivatives (gradient + Hessian)
//! 3. Solve Newton step: Δp = -H⁻¹g
//! 4. Update pose and check convergence
//!
//! Based on Magnusson 2009, Chapter 6.

pub mod async_pipeline;
pub mod batch_pipeline;
pub mod debug;
pub mod full_gpu_pipeline_v2;
pub mod gpu_initial_pose;
pub mod gpu_newton;
pub mod line_search;
pub mod more_thuente;
pub mod newton;
pub mod oscillation;
pub mod regularization;
pub mod solver;
pub mod types;

#[cfg(feature = "debug-iteration")]
pub use debug::IterationDebug;
pub use debug::{AlignmentDebug, AlignmentTimingDebug, IterationTimingDebug};
pub use full_gpu_pipeline_v2::{FullGpuOptimizationResultV2, FullGpuPipelineV2, PipelineV2Config};
pub use gpu_initial_pose::{
    BatchedNdtResult, GpuInitialPoseConfig, GpuInitialPosePipeline, PipelineMemoryRequirements,
};
pub use gpu_newton::{GpuNewtonError, GpuNewtonSolver};
pub use line_search::{LineSearchConfig, LineSearchResult};
pub use more_thuente::{MoreThuenteConfig, MoreThuenteResult, more_thuente_search};
pub use newton::{newton_step, newton_step_regularized};
pub use oscillation::{
    DEFAULT_OSCILLATION_THRESHOLD, OscillationResult, count_oscillation,
    count_oscillation_from_arrays,
};
pub use regularization::{RegularizationConfig, RegularizationTerm};
pub use solver::{NdtOptimizer, OptimizationConfig};
pub use types::{ConvergenceStatus, NdtConfig, NdtResult};

// Batch processing pipeline for parallel multi-scan alignment
pub use batch_pipeline::{
    AlignmentRequest, BatchAlignmentResult, BatchGpuPipeline, BatchPipelineConfig,
};

// Async batch pipeline with double buffering (Phase 23.1)
pub use async_pipeline::AsyncBatchPipeline;
