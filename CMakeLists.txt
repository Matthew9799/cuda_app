cmake_minimum_required(VERSION 3.10)
project(cuda)

set(
        CMAKE_CXX_FLAGS -fPIC
)

find_package(CUDA QUIET REQUIRED)

include_directories(
        include
        src
)

set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
)

cuda_add_executable(
        cuda_app
        main.cpp

)
