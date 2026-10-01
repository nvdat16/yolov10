"""Build configuration for the local YOLOv10 package and its CUDA extension."""

import os
import warnings
from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).parent.resolve()
MSDA_ROOT = ROOT / "ultralytics" / "nn" / "ops" / "ms_deform_attn"


def build_msda_extension():
    """Return the bundled Multi-Scale Deformable Attention CUDA extension when CUDA is available."""
    build_setting = os.getenv("YOLOV10_BUILD_MSDA", "auto").lower()
    if build_setting in {"0", "false", "no", "off"}:
        return [], {}

    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
    except ImportError as error:
        if build_setting in {"1", "true", "yes", "on"}:
            raise RuntimeError(
                "PyTorch must be installed before building MultiScaleDeformableAttention. "
                "Install PyTorch first, then run `pip install -e . --no-build-isolation`."
            ) from error
        warnings.warn(
            "PyTorch is not available in the isolated build environment; "
            "skipping MultiScaleDeformableAttention. Install with "
            "`pip install -e . --no-build-isolation` to build the CUDA extension."
        )
        return [], {}

    if CUDA_HOME is None:
        if build_setting in {"1", "true", "yes", "on"}:
            raise RuntimeError("CUDA toolkit was not found; cannot build MultiScaleDeformableAttention.")
        warnings.warn("CUDA toolkit was not found; skipping MultiScaleDeformableAttention.")
        return [], {}

    source_root = MSDA_ROOT / "src"
    sources = [
        source_root / "vision.cpp",
        source_root / "cpu" / "ms_deform_attn_cpu.cpp",
        source_root / "cuda" / "ms_deform_attn_cuda.cu",
    ]
    extension = CUDAExtension(
        name="ultralytics.nn.MultiScaleDeformableAttention",
        sources=[str(path.relative_to(ROOT)) for path in sources],
        include_dirs=[str(source_root)],
        define_macros=[("WITH_CUDA", None)],
        extra_compile_args={
            "cxx": ["-O2"],
            "nvcc": [
                "-O2",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        },
    )
    return [extension], {"build_ext": BuildExtension}


ext_modules, cmdclass = build_msda_extension()
setup(ext_modules=ext_modules, cmdclass=cmdclass)
