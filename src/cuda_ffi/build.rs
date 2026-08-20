//! Build script for cuda_ffi crate.
//!
//! Compiles CUDA code using nvcc and links against CUDA runtime.
//! Compilation is parallelized using rayon's work-stealing thread pool.

use rayon::prelude::*;
use std::{
    collections::BTreeSet,
    env,
    path::{Path, PathBuf},
    process::Command,
};

/// Architectures a build carries SASS for by default.
///
/// One entry per card this project is expected to run on, because a kernel
/// image is only usable by the architecture it was built for:
///
/// | arch   | hardware                                    |
/// |--------|---------------------------------------------|
/// | sm_86  | RTX 3090, A10                               |
/// | sm_87  | Jetson AGX Orin, the vehicle's computer     |
/// | sm_89  | RTX 4090, L40S                              |
/// | sm_90  | H100                                        |
/// | sm_120 | RTX 5090 and the rest of consumer Blackwell |
///
/// Cards newer than the toolkit are covered by the PTX embedded alongside
/// these, which the driver compiles on first launch. Older cards are not
/// supported, and fail loudly rather than silently.
const DEPLOY_ARCHES: &[u32] = &[86, 87, 89, 90, 120];

fn main() {
    // Which architectures to build for, and which toolkit can do it. Order
    // matters: the requested set picks the toolkit, not the other way round,
    // because a host may have several toolkits installed and only the newest
    // knows the newest cards.
    let requested = requested_arches();
    let toolkit = select_toolkit(&requested);
    let arches = if requested.is_empty() {
        // CUDA_ARCHS=all
        toolkit.arches.iter().copied().collect()
    } else {
        toolkit.filter_supported(&requested)
    };

    let cuda_include = toolkit.root.join("include");
    let cuda_lib = toolkit.root.join("lib64");

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

    for var in ["CUDA_ARCH", "CUDA_ARCHS", "CUDA_PATH", "CUDA_HOME"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!(
        "cargo:warning=building CUDA kernels with {} for sm_{}",
        toolkit.root.display(),
        join_arches(&arches)
    );

    let gencodes = gencode_flags(&arches);

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
        compile_cuda_source(source, &out_dir, &cuda_include, &toolkit.nvcc, &gencodes);
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

/// A CUDA toolkit found on this host, and the architectures its nvcc can target.
struct Toolkit {
    root: PathBuf,
    nvcc: PathBuf,
    arches: BTreeSet<u32>,
}

impl Toolkit {
    /// Read a toolkit's architecture list, or None if there is no usable nvcc.
    ///
    /// `nvcc --list-gpu-arch` is the authority. Inferring support from the
    /// directory name would be guesswork, and guessing wrong produces
    ///
    ///     nvcc fatal : Value 'sm_120' is not defined for option 'gpu-architecture'
    ///
    /// which names a flag rather than the mismatch behind it.
    fn probe(root: PathBuf) -> Option<Self> {
        let nvcc = root.join("bin").join("nvcc");
        if !nvcc.exists() {
            return None;
        }
        let out = Command::new(&nvcc).arg("--list-gpu-arch").output().ok()?;
        if !out.status.success() {
            return None;
        }
        let arches: BTreeSet<u32> = String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|line| line.trim().strip_prefix("compute_"))
            .filter_map(|digits| digits.parse().ok())
            .collect();
        if arches.is_empty() {
            return None;
        }
        Some(Self { root, nvcc, arches })
    }

    fn filter_supported(&self, requested: &[u32]) -> Vec<u32> {
        let (ok, dropped): (Vec<u32>, Vec<u32>) = requested
            .iter()
            .copied()
            .partition(|arch| self.arches.contains(arch));
        if !dropped.is_empty() {
            // Not fatal: a Jetson has one toolkit and cannot build for cards it
            // will never meet. Say so, because the resulting binary silently
            // lacks images for those architectures.
            println!(
                "cargo:warning={} cannot target sm_{}; cards newer than this toolkit \
                 fall back to the embedded PTX, older ones fail with \
                 cudaErrorNoKernelImageForDevice",
                self.root.display(),
                join_arches(&dropped)
            );
        }
        if ok.is_empty() {
            panic!(
                "{} supports none of the requested architectures {requested:?}",
                self.root.display()
            );
        }
        ok
    }
}

fn join_arches(arches: &[u32]) -> String {
    arches
        .iter()
        .map(|arch| arch.to_string())
        .collect::<Vec<_>>()
        .join(", sm_")
}

/// Every CUDA toolkit on this host: the one named by the environment first,
/// then the rest newest first.
///
/// CUDA_PATH and CUDA_HOME lead the list rather than replacing it. They are
/// commonly left pointing at whatever was installed first -- often the
/// unversioned `/usr/local/cuda` symlink, which follows the system
/// alternatives and on a machine with three toolkits installed is as likely to
/// be the oldest as the newest. Honouring such a variable absolutely would
/// silently drop the architectures it cannot build, which is the failure this
/// whole selection exists to prevent. A pin that can satisfy the request is
/// still used, because it wins every tie.
fn discover_toolkits() -> Vec<Toolkit> {
    let mut roots: Vec<PathBuf> = Vec::new();
    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(path) = env::var(var) {
            roots.push(PathBuf::from(path));
        }
    }

    let mut versioned: Vec<(Vec<u32>, PathBuf)> = std::fs::read_dir("/usr/local")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            let version = name.strip_prefix("cuda-")?;
            // A bare "cuda-12" is a floating alias for the newest 12.x and adds
            // nothing this loop does not learn from the real directory.
            let parts: Vec<u32> = version.split('.').map(|p| p.parse().unwrap_or(0)).collect();
            (parts.len() >= 2).then_some((parts, path))
        })
        .collect();
    versioned.sort_by(|a, b| b.0.cmp(&a.0));

    roots.extend(versioned.into_iter().map(|(_, path)| path));
    roots.extend([PathBuf::from("/usr/local/cuda"), PathBuf::from("/opt/cuda")]);

    let mut seen = BTreeSet::new();
    roots
        .into_iter()
        .filter(|root| seen.insert(std::fs::canonicalize(root).unwrap_or_else(|_| root.clone())))
        .filter_map(Toolkit::probe)
        .collect()
}

/// The toolkit that covers the most of what was asked for.
///
/// Ties go to the earliest candidate, so an adequate CUDA_PATH is honoured and
/// otherwise the newest toolkit wins. Choosing by coverage is what lets a host
/// whose `/usr/local/cuda` still points at 12.3 build Blackwell images from the
/// 12.8 it also has installed.
fn select_toolkit(requested: &[u32]) -> Toolkit {
    let toolkits = discover_toolkits();
    if toolkits.is_empty() {
        panic!("CUDA installation not found. Set CUDA_PATH or CUDA_HOME environment variable.");
    }

    let coverage = |toolkit: &Toolkit| {
        requested
            .iter()
            .filter(|arch| toolkit.arches.contains(arch))
            .count()
    };
    // Reverse on the index because max_by_key keeps the LAST maximum, and the
    // tie has to go to the earliest candidate for CUDA_PATH to mean anything.
    let best = toolkits
        .iter()
        .enumerate()
        .max_by_key(|(index, toolkit)| (coverage(toolkit), std::cmp::Reverse(*index)))
        .map(|(index, _)| index)
        .expect("non-empty");

    if best != 0 {
        // Loud, because it means the environment's CUDA is not the one being
        // compiled against. Narrowing the request puts it back.
        let missing: Vec<u32> = requested
            .iter()
            .copied()
            .filter(|arch| !toolkits[0].arches.contains(arch))
            .collect();
        println!(
            "cargo:warning=using {} instead of {}, which cannot build sm_{}; set \
             CUDA_ARCHS=native to build only for this machine's GPU",
            toolkits[best].root.display(),
            toolkits[0].root.display(),
            join_arches(&missing)
        );
    }

    toolkits.into_iter().nth(best).expect("index is in range")
}

/// Architectures to build for.
///
/// `CUDA_ARCHS` accepts a comma-separated list, or one of:
///
/// - `deploy` (the default) -- every card this project runs on, see DEPLOY_ARCHES
/// - `native` -- only what is in this machine, which is much faster to build
/// - `all` -- everything the chosen toolkit can emit
///
/// `CUDA_ARCH` (singular) remains as it was, naming exactly one architecture.
fn requested_arches() -> Vec<u32> {
    if let Ok(one) = env::var("CUDA_ARCH") {
        let arch = parse_arch(one.trim());
        return vec![arch];
    }

    match env::var("CUDA_ARCHS").as_deref().map(str::trim) {
        Err(_) | Ok("deploy") => DEPLOY_ARCHES.to_vec(),
        Ok("native") => vec![detect_cuda_arch()],
        // An empty request means "keep everything the toolkit supports", which
        // is resolved once the toolkit is known.
        Ok("all") => Vec::new(),
        Ok(list) => {
            let arches: Vec<u32> = list
                .split(',')
                .map(str::trim)
                .filter(|entry| !entry.is_empty())
                .map(parse_arch)
                .collect();
            if arches.is_empty() {
                panic!("CUDA_ARCHS is set but lists no architectures");
            }
            arches
        }
    }
}

/// Accept 87, sm_87 and compute_87 alike; they all name one architecture.
fn parse_arch(text: &str) -> u32 {
    text.trim_start_matches("sm_")
        .trim_start_matches("compute_")
        .parse()
        .unwrap_or_else(|_| panic!("{text:?} is not a CUDA architecture such as 87 or sm_87"))
}

/// nvcc flags that embed SASS for each architecture, plus PTX for the newest.
///
/// The PTX is what makes a card newer than this toolkit work at all: the driver
/// compiles it on first launch. Without it such a card gets no runnable image
/// and every launch returns cudaErrorNoKernelImageForDevice -- CUDA error 209,
/// which surfaces only as "NDT alignment failed: CUDA error code 209" in the
/// node's stderr. The node otherwise looks healthy, keeps its services and
/// reports "Node is not activated", so the visible symptom is that localization
/// never initializes, with nothing pointing at the GPU.
fn gencode_flags(arches: &[u32]) -> Vec<String> {
    let mut sorted: Vec<u32> = arches.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    let newest = *sorted.last().expect("at least one architecture");
    let mut flags: Vec<String> = sorted
        .iter()
        .map(|arch| format!("-gencode=arch=compute_{arch},code=sm_{arch}"))
        .collect();
    flags.push(format!(
        "-gencode=arch=compute_{newest},code=compute_{newest}"
    ));
    flags
}

/// Compile a CUDA source file using nvcc.
fn compile_cuda_source(
    source: &str,
    out_dir: &Path,
    cuda_include: &Path,
    nvcc: &Path,
    gencodes: &[String],
) {
    let source_path = PathBuf::from(source);
    let stem = source_path.file_stem().unwrap().to_str().unwrap();
    let obj_path = out_dir.join(format!("{stem}.o"));
    let lib_path = out_dir.join(format!("lib{stem}.a"));

    let mut args: Vec<String> = vec![
        "-c".into(),
        "-o".into(),
        obj_path.to_str().unwrap().into(),
        source.into(),
        "-I".into(),
        cuda_include.to_str().unwrap().into(),
        // Generate position-independent code for shared library
        "-Xcompiler".into(),
        "-fPIC".into(),
        // Optimize
        "-O3".into(),
    ];
    // No -arch alongside these: it names a single target and would contradict
    // the list.
    args.extend(gencodes.iter().cloned());

    // Compile with nvcc
    let output = Command::new(nvcc)
        .args(&args)
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

/// Compute capability of the first local GPU as nvcc wants it (86 for 8.6).
///
/// Falls back to the Jetson Orin's 87 when there is no usable GPU here, so a
/// cross-build or a CI box without CUDA still produces the vehicle's target.
fn detect_cuda_arch() -> u32 {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output();
    match out {
        Ok(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout);
            let first = text.lines().next().unwrap_or("").trim();
            let digits: String = first.chars().filter(|c| c.is_ascii_digit()).collect();
            match digits.parse() {
                Ok(arch) => arch,
                Err(_) => {
                    println!("cargo:warning=could not parse compute_cap {first:?}, using sm_87");
                    87
                }
            }
        }
        _ => {
            println!("cargo:warning=nvidia-smi unavailable, building for Jetson Orin sm_87");
            87
        }
    }
}
