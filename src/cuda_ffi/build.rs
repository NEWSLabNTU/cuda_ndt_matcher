//! Build script for cuda_ffi crate.
//!
//! Compiles CUDA code using nvcc and links against CUDA runtime.
//! Compilation is parallelized using rayon's work-stealing thread pool.

use rayon::prelude::*;
use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

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

    // Determine target architecture.
    //
    // CUDA_ARCH wins; otherwise ask the local GPU; otherwise fall back to the
    // Jetson Orin's sm_87, which is what this package targets on the vehicle.
    //
    // Detecting matters because the failure is remote from the cause. Kernels
    // built for sm_87 do not run on, say, an sm_86 workstation GPU: every launch
    // returns CUDA error 209, cudaErrorNoKernelImageForDevice, which surfaces
    // only as "NDT alignment failed: CUDA error code 209" in the node's stderr.
    // The node otherwise looks healthy -- it keeps its services, publishes
    // diagnostics, and reports "Node is not activated" -- so the visible symptom
    // is that localization never initializes, with nothing pointing at the GPU.
    let cuda_arch = clamp_to_supported_arch(
        env::var("CUDA_ARCH").unwrap_or_else(|_| detect_cuda_arch()),
    );
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:warning=building CUDA kernels for sm_{cuda_arch}");

    // Compile CUDA source files
    let cuda_sources = [
        "csrc/radix_sort.cu",
        "csrc/segment_detect.cu",
        "csrc/segmented_reduce.cu",
        "csrc/batched_solve.cu",
        "csrc/voxel_hash.cu",
        "csrc/batch_persistent_ndt.cu",
        "csrc/async_stream.cu",
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
    println!("cargo:rustc-link-lib=static=batch_persistent_ndt");
    println!("cargo:rustc-link-lib=static=async_stream");
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

    // Build architecture flags.
    //
    // Both SASS and PTX are embedded. The SASS is what runs on a GPU this
    // toolkit knows; the PTX is what lets the driver JIT for a newer one, which
    // is the case when the card is ahead of the CUDA release — an RTX 5090
    // (sm_120) against CUDA 12.3, whose newest target is sm_90. Without the PTX
    // that pairing produces no runnable image and every launch returns
    // cudaErrorNoKernelImageForDevice.
    let arch_flag = format!("-arch=sm_{cuda_arch}");
    let gencode_flag =
        format!("-gencode=arch=compute_{cuda_arch},code=[sm_{cuda_arch},compute_{cuda_arch}]");

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

/// Compute capability of the first local GPU as nvcc wants it ("86" for 8.6).
///
/// Falls back to the Jetson Orin's "87" when there is no usable GPU here, so a
/// cross-build or a CI box without CUDA still produces the vehicle's target.
/// Architectures this nvcc can actually target, newest first.
///
/// Empty when `nvcc --list-gpu-arch` is unavailable or unparseable, which the
/// caller treats as "do not clamp" rather than as an error: an older nvcc
/// without the flag should not stop a build that would otherwise work.
fn supported_arches() -> Vec<u32> {
    let out = std::process::Command::new("nvcc")
        .arg("--list-gpu-arch")
        .output();
    let Ok(out) = out else { return Vec::new() };
    if !out.status.success() {
        return Vec::new();
    }
    let mut arches: Vec<u32> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| line.trim().strip_prefix("compute_"))
        .filter_map(|digits| digits.parse::<u32>().ok())
        .collect();
    arches.sort_unstable();
    arches.reverse();
    arches
}

/// Reduce a requested arch to the newest one this nvcc supports.
///
/// Detecting the local GPU is not enough on its own: a card can be newer than
/// the installed CUDA. nvcc then rejects the arch outright —
///
///   nvcc fatal : Value 'sm_120' is not defined for option 'gpu-architecture'
///
/// — which stops the build with a message about a flag rather than about the
/// mismatch that caused it. Clamping keeps the build working, and the PTX
/// embedded alongside the SASS is what makes the result run on the newer card.
fn clamp_to_supported_arch(requested: String) -> String {
    let Ok(requested_num) = requested.parse::<u32>() else {
        return requested;
    };
    let supported = supported_arches();
    if supported.is_empty() || supported.contains(&requested_num) {
        return requested;
    }
    match supported.iter().find(|&&arch| arch <= requested_num) {
        Some(&arch) => {
            println!(
                "cargo:warning=this nvcc cannot target sm_{requested_num}; building for \
                 sm_{arch} with PTX, which the driver will JIT for the local GPU"
            );
            arch.to_string()
        }
        None => {
            // The GPU is older than anything this toolkit emits. Nothing to
            // fall back to, so leave it and let nvcc give its own message.
            requested
        }
    }
}

fn detect_cuda_arch() -> String {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output();
    match out {
        Ok(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout);
            let first = text.lines().next().unwrap_or("").trim();
            let digits: String = first.chars().filter(|c| c.is_ascii_digit()).collect();
            if digits.is_empty() {
                println!("cargo:warning=could not parse compute_cap {first:?}, using sm_87");
                "87".to_string()
            } else {
                digits
            }
        }
        _ => {
            println!("cargo:warning=nvidia-smi unavailable, building for Jetson Orin sm_87");
            "87".to_string()
        }
    }
}
