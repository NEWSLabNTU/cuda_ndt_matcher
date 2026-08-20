#!/usr/bin/env bash
# Point CUDA_PATH and LD_LIBRARY_PATH at a toolkit that can build for the local
# GPU. Source this; it does not run standalone.
#
# CubeCL compiles its kernels at runtime with NVRTC, using the compute
# capability it reads from the device and no fallback. When the GPU is newer
# than the default toolkit -- an RTX 5090 at compute 12.0 against CUDA 12.3,
# whose NVRTC tops out below that -- every kernel launch dies with
#
#     nvrtc: error: invalid value for --gpu-architecture (-arch)
#
# which surfaces as a wall of failing GPU tests that say nothing about the
# toolkit. The ahead-of-time kernels in cuda_ffi clamp their arch and embed PTX
# instead (see src/cuda_ffi/build.rs), but nothing in this process can clamp
# what CubeCL does at runtime, so the answer here is to hand it a newer NVRTC.
#
# Hosts whose default toolkit already covers the GPU -- the Jetson AGX Orin at
# compute 8.7 on CUDA 12.6, for one -- keep whatever the environment already
# had. This is deliberately confined to this crate's own recipes: the rest of
# the workspace links against TIER IV's prebuilt CUDA and TensorRT libraries,
# and moving the whole workspace onto a different toolkit is a much larger
# change than making these tests run.

select_cuda() {
    local cap
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
        head -1 | tr -d ' .')
    if [[ -z "$cap" ]]; then
        return 0
    fi

    # Newest first, so the chosen toolkit is the most capable one that works.
    local root
    for root in $(ls -d /usr/local/cuda-*.* 2>/dev/null | sort -Vr); do
        [[ -x "$root/bin/nvcc" ]] || continue
        if "$root/bin/nvcc" --list-gpu-arch 2>/dev/null |
            grep -qx "compute_${cap}"; then
            export CUDA_PATH="$root"
            export LD_LIBRARY_PATH="$root/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            echo "Using CUDA at $root for compute_${cap}"
            return 0
        fi
    done

    echo "warning: no installed CUDA toolkit targets compute_${cap};" \
        "CubeCL kernels will fail to compile at runtime" >&2
}

select_cuda
