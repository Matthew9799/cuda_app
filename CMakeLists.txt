cmake_minimum_required(VERSION 3.10)
project(cuda)

set(CMAKE_CONFIGURATION_TYPES "Debug" CACHE STRING "" FORCE)

configure_file(matt.JPG matt.JPG COPYONLY)

if(UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS -fPIC)
elseif(WIN32)
    set(OpenCV_DIR "C:\\Program Files\\openCV\\openCV\\opencv\\build")
    set(CMAKE_PREFIX_PATH "C:\\Qt\\5.12.6\\msvc2017_64")
endif(UNIX AND NOT APPLE)

find_package(CUDA QUIET)
find_package(OpenCV QUIET REQUIRED)
find_package(OpenCL QUIET)
find_package(OpenMP QUIET REQUIRED)

include_directories(
        include
        src
        ${OpenCV_INCLUDE_DIRS}
)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

if(CUDA_FOUND)
    message("-- Cuda found")

    set(
            CUDA_NVCC_FLAGS
            "${CUDA_NVCC_FLAGS}"
    )

    add_definitions(-Dcuda_compile_check)

    cuda_add_executable(
            cuda_app
            main.cpp
            src/cuda_kernels.cu
            src/helper.cpp
            include/cuda_kernels.hpp
            include/accelerator.hpp
            include/helper.hpp
    )

    target_link_libraries(cuda_app ${CUDA_LIBRARIES})
elseif(OpenCL_FOUND)
    message("-- OpenCL found")

    configure_file( src/opencl_kernels.cl opencl_kernels.cl COPYONLY)

    include_directories(${OpenCL_INCLUDE_DIRS})

    LINK_DIRECTORIES(${OpenCL_LIBRARY})

    add_definitions(-Dopencl_compile_check)

    add_executable(
            cuda_app
            main.cpp
            src/opencl_kernels.cpp
            src/helper.cpp
            include/opencl_kernels.hpp
            include/accelerator.hpp
            include/helper.hpp
    )

    target_link_libraries(cuda_app ${OpenCL_LIBRARY})
elseif(OpenMP_FOUND)
    message("-- Nothing found")

    add_executable(
            cuda_app
            main.cpp
            src/cpu_kernels.cpp
            src/helper.cpp
            include/accelerator.hpp
            include/cpu_kernels.hpp
            include/helper.hpp
    )

endif()

target_link_libraries(cuda_app ${OpenCV_LIBS})
