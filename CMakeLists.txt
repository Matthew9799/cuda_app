cmake_minimum_required(VERSION 3.10)
project(cuda)

if(UNIX)
    set(CMAKE_CXX_FLAGS -fPIC)
endif(UNIX)

find_package(CUDA QUIET REQUIRED)

find_package(JUCE REQUIRED
        COMPONENTS
        juce_core
        juce_data_structures
        juce_events
        juce_graphics
        juce_gui_basics
        juce_gui_extra
)

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
        src/kernels.cu
        include/kernels.h
)
