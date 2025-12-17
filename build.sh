#!/bin/bash

set -e  # stop on any error

BUILD_DIR="build"
CU_DIR="cu_files"

# 1. Create build directory or clear it
if [ -d "$BUILD_DIR" ]; then
    echo "[INFO] build/ exists. Clearing contents..."
    rm -rf "$BUILD_DIR"/*
else
    echo "[INFO] Creating build/ directory..."
    mkdir "$BUILD_DIR"
fi

# 2. List of CUDA source files to compile
CU_FILES=(
    "serial.cu"
    "hierarchical.cu"
    "optimal.cu"
    "shared_memory.cu"
    "thread_coarsening.cu"
)

echo "[INFO] Compiling CUDA files..."

# Compile each file
for cu in "${CU_FILES[@]}"; do
    src="${CU_DIR}/${cu}"
    exe="${BUILD_DIR}/${cu%.cu}"   # remove .cu extension
    echo "  → nvcc $src -o $exe"
    nvcc "$src" -o "$exe"
done

# cutlass version compile
nvcc -O3 -std=c++17 ${CU_DIR}/cutlass.cu -I ./cutlass/include -I ./cutlass/tools/util/include -arch=sm_75 -o ${BUILD_DIR}/cutlass

echo "[INFO] Build complete! Executables are in $BUILD_DIR/"
