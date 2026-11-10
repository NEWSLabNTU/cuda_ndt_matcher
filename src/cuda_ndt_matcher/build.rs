//! Points the node at an NVRTC that can compile CubeCL's kernels for the local
//! GPU, whatever library path it is launched with. The logic, and why it has
//! to happen at link time, is in `ndt_cuda/nvrtc_select.rs`.

#[path = "../ndt_cuda/nvrtc_select.rs"]
mod nvrtc_select;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../ndt_cuda/nvrtc_select.rs");
    nvrtc_select::emit_rpath();
}
