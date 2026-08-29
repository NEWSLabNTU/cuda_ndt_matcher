//! Pins the CUDA version that `cubecl-cuda` compiles against.
//!
//! This build script does nothing itself. It exists so that the
//! `[build-dependencies]` block in Cargo.toml is actually resolved, which is
//! what forces `cubecl-cuda`'s own build-dependency copy of `cudarc` onto an
//! explicit CUDA floor.
//!
//! Why that matters is recorded at the `cudarc017` pin in Cargo.toml. The short
//! version: `cubecl-cuda`'s build script branches on
//! `cudarc::driver::sys::CUDA_VERSION`, read from its *build-dependency* copy
//! of cudarc, which asks `nvcc --version` and silently assumes CUDA 13.0 when
//! nvcc is off PATH. That guess decides whether `cubecl-cuda`'s source
//! references the CUDA 12.8 tensormap API, while the *normal* copy of cudarc --
//! the one that decides whether those symbols exist -- is pinned here. When the
//! two disagree the crate does not compile.
//!
//! Cargo's v2 resolver keeps build-dependency features separate from normal
//! ones, so pinning the normal copy cannot reach the build-dependency copy.
//! Declaring the same crate as a build-dependency here is what closes that gap.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
}
