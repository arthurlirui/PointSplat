#!/bin/bash
# Environment setup for PointSplat - configures MSVC + CUDA for gsplat JIT compilation

# MSVC 2022 paths
MSVC_ROOT="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.38.33130"
WIN_SDK_INC="C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0"
WIN_SDK_LIB="C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22621.0"
CUDA_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"

# Add to PATH
export PATH="D:/conda-envs/pointsplat/Scripts:${MSVC_ROOT}/bin/Hostx64/x64:${CUDA_ROOT}/bin:${CUDA_ROOT}/libnvvp:$PATH"

# Set CUDA_HOME
export CUDA_HOME="${CUDA_ROOT}"

# Set include paths for cl.exe
export INCLUDE="${MSVC_ROOT}/include;${WIN_SDK_INC}/ucrt;${WIN_SDK_INC}/um;${WIN_SDK_INC}/shared;${CUDA_ROOT}/include"
export LIB="${MSVC_ROOT}/lib/x64;${WIN_SDK_LIB}/ucrt/x64;${WIN_SDK_LIB}/um/x64;${CUDA_ROOT}/lib/x64"

# Tell torch to use cl.exe
export CC=cl
export CXX=cl

# Set CUDA arch for RTX A4500 (Ampere = 8.6)
export TORCH_CUDA_ARCH_LIST="8.6"

# Disable triton check (not needed for inference)
export XFORMERS_NO_TRITION=1
