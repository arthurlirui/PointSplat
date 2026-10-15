"""Set up the build environment for gsplat CUDA JIT compilation on Windows.

This module sets the environment variables needed for torch's cpp_extension
to find MSVC (cl.exe), CUDA (nvcc), and ninja, so that gsplat can compile
its CUDA backend on first import.

Usage: import this module before importing gsplat or torch extensions.
    import scripts.setup_build_env  # noqa: F401
    from gsplat.cuda._backend import _C
"""

import os
import shutil
from pathlib import Path


def _setup_build_env():
    """Configure environment variables for CUDA extension compilation."""
    # Conda environment paths
    conda_scripts = r"D:\conda-envs\pointsplat\Scripts"

    # MSVC 2022 paths
    msvc_bin = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64"
    msvc_lib = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\lib\x64"
    msvc_inc = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\include"

    # Windows SDK
    win_sdk_inc = r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0"
    win_sdk_lib = r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0"

    # CUDA toolkit
    cuda_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

    # Build PATH (prepend to existing)
    extra_paths = [
        conda_scripts,
        msvc_bin,
        os.path.join(cuda_root, "bin"),
        os.path.join(cuda_root, "libnvvp"),
    ]
    existing_path = os.environ.get("PATH", "")
    # Use os.pathsep (; on Windows) to join
    os.environ["PATH"] = os.pathsep.join(extra_paths) + os.pathsep + existing_path

    # CUDA_HOME
    os.environ["CUDA_HOME"] = cuda_root

    # INCLUDE paths for cl.exe
    includes = [
        msvc_inc,
        os.path.join(win_sdk_inc, "ucrt"),
        os.path.join(win_sdk_inc, "um"),
        os.path.join(win_sdk_inc, "shared"),
        os.path.join(cuda_root, "include"),
    ]
    os.environ["INCLUDE"] = os.pathsep.join(includes)

    # LIB paths for linker
    libs = [
        msvc_lib,
        os.path.join(win_sdk_lib, "ucrt", "x64"),
        os.path.join(win_sdk_lib, "um", "x64"),
        os.path.join(cuda_root, "lib", "x64"),
    ]
    os.environ["LIB"] = os.pathsep.join(libs)

    # Compiler
    os.environ["CC"] = "cl"
    os.environ["CXX"] = "cl"

    # CUDA arch for RTX A4500 (Ampere = 8.6)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"

    # Patch distutils._msvccompiler for setuptools 80+ / Python 3.10+
    # torch's cpp_extension calls distutils._msvccompiler._get_vc_env(),
    # but setuptools moved it to setuptools._distutils._msvccompiler.
    try:
        import distutils
        if not hasattr(distutils, "_msvccompiler"):
            from setuptools._distutils import _msvccompiler as _msvc
            distutils._msvccompiler = _msvc
    except Exception:
        pass


_setup_build_env()
