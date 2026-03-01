# Phase 26: Migrate to cargo-nextest

**Status**: Complete
**Date**: 2026-02-27

## Motivation

The project uses `cargo test` for 495 Rust tests across 3 crates and `pytest` for 13 Python integration tests. Current problems:

1. **GPU test flakiness**: `cargo test` runs tests as threads in one process — shared CubeCL/CUDA state causes 3 tests to be permanently `#[ignore]`d
2. **No structured output**: Plain text output; no JUnit XML for CI
3. **No timeout enforcement**: Rust tests can hang indefinitely
4. **No parallelism control**: Cannot serialize GPU-dependent tests while keeping CPU tests parallel

## Goals

- Adopt `cargo-nextest` as the test runner for all Rust tests
- Serialize GPU tests via test groups to eliminate flakiness
- Produce JUnit XML reports for CI integration
- Un-ignore previously-flaky GPU tests
- Keep Python integration tests via pytest (unchanged)

## Criteria

- [x] `.config/nextest.toml` created with `gpu` test group (`max-threads = 1`)
- [x] `gpu` group covers `ndt_cuda` and `cuda_ffi` packages (both use CUDA device)
- [x] Slow-timeout overrides: 120s for `ndt_cuda`, 60s for `cuda_ffi`
- [x] CI profile with `fail-fast = true`
- [x] JUnit XML output configured (`target/nextest/default/junit.xml`)
- [x] `require_cuda!()` macro reverted to `return` with `eprintln!` (not `panic!`)
- [x] 3 flaky `#[ignore]` tests un-ignored (`test_align_full_gpu_identity`, `test_align_full_gpu_vs_cpu`, `test_gpu_cpu_alignment_consistency`)
- [x] 2 GPU init pose tests un-ignored with `require_cuda!()` guard (`test_pipeline_creation`, `test_evaluate_batch_empty`)
- [x] `test_async_pipeline_poll` fixed — replaced spin loop with time-based deadline
- [x] Justfile `test-rust`, `test-ndt-cuda`, `test-cuda-ffi`, `test-cuda-ndt-matcher` recipes use `cargo nextest run`
- [x] `setup` recipe checks for `cargo-nextest`
- [x] 490 tests pass, 1 skipped (opt-in benchmark `test_align_performance`)
- [x] Previously-flaky tests pass reliably with nextest

## Files Changed

| File                                                | Change                                                                            |
|-----------------------------------------------------|-----------------------------------------------------------------------------------|
| `.config/nextest.toml`                              | Created — gpu test group, slow-timeout overrides, JUnit XML, CI profile           |
| `justfile`                                          | `cargo test` → `cargo nextest run` in 4 recipes; `cargo-nextest` check in `setup` |
| `src/ndt_cuda/src/runtime.rs`                       | `require_cuda!()` reverted from `panic!` to `return` + `eprintln!`                |
| `src/ndt_cuda/src/ndt.rs`                           | Same `require_cuda!()` revert; un-ignored `test_gpu_cpu_alignment_consistency`    |
| `src/ndt_cuda/src/optimization/solver.rs`           | Un-ignored `test_align_full_gpu_identity`, `test_align_full_gpu_vs_cpu`           |
| `src/ndt_cuda/src/optimization/gpu_initial_pose.rs` | Un-ignored 2 tests, added `require_cuda!()` guard                                 |
| `src/ndt_cuda/src/optimization/async_pipeline.rs`   | Fixed poll timeout: time-based deadline + `thread::yield_now()`                   |

## Verification

```bash
just test                    # All Rust tests via nextest
just test-integration        # Python tests unchanged
ls target/nextest/default/junit.xml  # JUnit output exists
```
