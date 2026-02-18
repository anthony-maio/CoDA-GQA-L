"""Build fused differential attention epilogue kernel.

Install:
    cd kernels/coda_diff_epilogue
    pip install -e .

Or from repo root:
    pip install -e kernels/coda_diff_epilogue

Auto-detects GPU compute capability and builds for it, plus fat binary
targets for common GPUs (T4, A100, 3090, 4090, H100, Blackwell).
"""

import os
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_gpu_arch():
    """Detect current GPU's compute capability via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, timeout=5,
        ).strip().split("\n")[0]
        major, minor = out.split(".")
        return int(major), int(minor)
    except Exception:
        return None


def get_gencode_flags():
    """Build gencode flags for NVCC.

    Strategy: fat binary with SASS for common targets + PTX for forward compat.
    If we detect the current GPU, include its exact arch too.
    """
    # Common targets: SASS for exact match (fastest load, no JIT)
    targets = [
        (75, "sm"),      # T4, Turing
        (80, "sm"),      # A100, A30
        (86, "sm"),      # RTX 3090, 3080, A40
        (89, "sm"),      # RTX 4090, 4080, L40
        (90, "sm"),      # H100, H200
    ]
    # PTX for forward compatibility (JIT-compiled at runtime)
    ptx_targets = [
        (80, "compute"),  # covers any 8.x we missed
        (90, "compute"),  # covers Blackwell (12.x), future 9.x+
    ]

    # Add detected GPU if not already covered
    gpu = get_gpu_arch()
    if gpu:
        arch = gpu[0] * 10 + gpu[1]
        known = {t[0] for t in targets}
        if arch not in known and arch >= 75:
            targets.append((arch, "sm"))

    flags = []
    for cap, code_type in targets:
        flags.append(f"-gencode=arch=compute_{cap},code={code_type}_{cap}")
    for cap, code_type in ptx_targets:
        flags.append(f"-gencode=arch=compute_{cap},code={code_type}_{cap}")

    return flags


setup(
    name="coda_kernels",
    version="0.1.0",
    packages=["coda_kernels"],
    package_dir={"coda_kernels": "torch-ext/coda_kernels"},
    ext_modules=[
        CUDAExtension(
            name="coda_kernels._ops",
            sources=[
                "torch-ext/torch_binding.cpp",
                "kernel_src/diff_epilogue.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"] + get_gencode_flags(),
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
