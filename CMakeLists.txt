cmake_minimum_required(VERSION 3.10)
project(cuda)

find_package(CUDA QUIET REQUIRED)

include_directories(
        include
        src
)

include_directories(Program Files\JUCE\modules)

set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};

)

cuda_add_executable(
        cuda_app
        main.cpp
        src/kernels.cu
        include/kernels.h
)
