"""Build fused differential attention epilogue kernel."""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_75,code=sm_75",   # T4
                    "-gencode=arch=compute_80,code=sm_80",   # A100
                    "-gencode=arch=compute_90,code=sm_90",   # H100
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
