//! Build script for cuda_ffi crate.
//!
//! Compiles CUDA code using nvcc and links against CUDA runtime.
//! Compilation is parallelized using rayon's work-stealing thread pool.

use rayon::prelude::*;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Find CUDA installation
    let cuda_path = find_cuda_path();
    let cuda_include = cuda_path.join("include");
    let cuda_lib = cuda_path.join("lib64");

    // Find CUB headers (included with CUDA 11+)
    // CUB is header-only and included in CUDA toolkit
    if !cuda_include.join("cub").exists() {
        panic!(
            "CUB headers not found in {:?}. CUB is included with CUDA 11+.",
            cuda_include
        );
    }

    // Output directory
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Determine target architecture
    // Default to Jetson Orin (sm_87), can be overridden via CUDA_ARCH env var
    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "87".to_string());
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");

    // Compile CUDA source files
    let cuda_sources = [
        "csrc/radix_sort.cu",
        "csrc/segment_detect.cu",
        "csrc/segmented_reduce.cu",
        "csrc/batched_solve.cu",
        "csrc/voxel_hash.cu",
        "csrc/persistent_ndt.cu",
        "csrc/batch_persistent_ndt.cu",
        "csrc/async_stream.cu",
        "csrc/texture_voxels.cu",
        "csrc/ndt_graph_kernels.cu",
    ];

    // Compile in parallel using rayon (work-stealing thread pool)
    cuda_sources.par_iter().for_each(|source| {
        compile_cuda_source(source, &out_dir, &cuda_include, &cuda_arch);
    });

    // Link against CUDA runtime and cuSOLVER
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cusolver");

    // Link against C++ standard library (CUB uses C++ features)
    println!("cargo:rustc-link-lib=stdc++");

    // Link our compiled object files
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=radix_sort");
    println!("cargo:rustc-link-lib=static=segment_detect");
    println!("cargo:rustc-link-lib=static=segmented_reduce");
    println!("cargo:rustc-link-lib=static=batched_solve");
    println!("cargo:rustc-link-lib=static=voxel_hash");
    println!("cargo:rustc-link-lib=static=persistent_ndt");
    println!("cargo:rustc-link-lib=static=batch_persistent_ndt");
    println!("cargo:rustc-link-lib=static=async_stream");
    println!("cargo:rustc-link-lib=static=texture_voxels");
    println!("cargo:rustc-link-lib=static=ndt_graph_kernels");

    // Link these again after static libs to resolve symbols
    // (linker is single-pass, so static libs need symbols from these)
    println!("cargo:rustc-link-lib=cusolver"); // batched_solve needs cusolverDn*
    println!("cargo:rustc-link-lib=stdc++"); // CUB needs C++ runtime

    // Rerun if CUDA sources change
    for source in &cuda_sources {
        println!("cargo:rerun-if-changed={source}");
    }
    // Also watch header files
    println!("cargo:rerun-if-changed=csrc/persistent_ndt_device.cuh");
    println!("cargo:rerun-if-changed=csrc/cholesky_6x6.cuh");
    println!("cargo:rerun-if-changed=csrc/jacobi_svd_6x6.cuh");
    println!("cargo:rerun-if-changed=csrc/batch_persistent_ndt_device.cuh");
    println!("cargo:rerun-if-changed=csrc/warp_reduce.cuh");
    println!("cargo:rerun-if-changed=csrc/warp_cholesky.cuh");
    println!("cargo:rerun-if-changed=csrc/ndt_graph_common.cuh");
    println!("cargo:rerun-if-changed=build.rs");
}

/// Find CUDA installation path.
fn find_cuda_path() -> PathBuf {
    // Try environment variable first
    if let Ok(path) = env::var("CUDA_PATH") {
        return PathBuf::from(path);
    }
    if let Ok(path) = env::var("CUDA_HOME") {
        return PathBuf::from(path);
    }

    // Try common installation paths
    let common_paths = ["/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda"];

    for path in &common_paths {
        let p = PathBuf::from(path);
        if p.exists() {
            return p;
        }
    }

    panic!("CUDA installation not found. Set CUDA_PATH or CUDA_HOME environment variable.");
}

/// Compile a CUDA source file using nvcc.
///
/// The `cuda_arch` parameter specifies the compute capability (e.g., "87" for sm_87).
fn compile_cuda_source(source: &str, out_dir: &Path, cuda_include: &Path, cuda_arch: &str) {
    let source_path = PathBuf::from(source);
    let stem = source_path.file_stem().unwrap().to_str().unwrap();
    let obj_path = out_dir.join(format!("{stem}.o"));
    let lib_path = out_dir.join(format!("lib{stem}.a"));

    // Build architecture flags
    let arch_flag = format!("-arch=sm_{cuda_arch}");
    let gencode_flag = format!("-gencode=arch=compute_{cuda_arch},code=sm_{cuda_arch}");

    // Compile with nvcc
    let output = Command::new("nvcc")
        .args([
            "-c",
            "-o",
            obj_path.to_str().unwrap(),
            source,
            "-I",
            cuda_include.to_str().unwrap(),
            // Generate position-independent code for shared library
            "-Xcompiler",
            "-fPIC",
            // Optimize
            "-O3",
            // Target single architecture for faster compilation
            &arch_flag,
            &gencode_flag,
        ])
        .output()
        .expect("Failed to run nvcc. Is CUDA toolkit installed?");

    if !output.status.success() {
        panic!(
            "nvcc compilation failed for {source}:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // Create static library
    let output = Command::new("ar")
        .args([
            "rcs",
            lib_path.to_str().unwrap(),
            obj_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run ar");

    if !output.status.success() {
        panic!(
            "ar failed to create library for {stem}:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
