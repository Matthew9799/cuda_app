cmake_minimum_required(VERSION 3.10)
project(cuda)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_CONFIGURATION_TYPES "Debug" CACHE STRING "" FORCE)

configure_file(matt.JPG matt.JPG COPYONLY)

if(UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS -fPIC)
elseif(WIN32)
    set(OpenCV_DIR "C:\\Program Files\\openCV\\openCV\\opencv\\build")
    set(CMAKE_PREFIX_PATH "C:\\Qt\\5.12.6\\msvc2017_64")
endif(UNIX AND NOT APPLE)

find_package(CUDA QUIET REQUIRED)
find_package( OpenCV REQUIRED )
find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(OpenCL REQUIRED)

include_directories(
        include
        src
        ${OpenCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

LINK_DIRECTORIES(${OpenCL_LIBRARY})

set(
        CUDA_NVCC_FLAGS
        "${CUDA_NVCC_FLAGS} -lOpenCL"
)

cuda_add_executable(
        cuda_app
        main.cpp
        src/cuda_kernels.cu
        src/accelerator.cpp
        src/opencl_kernels.cpp
        include/cuda_kernels.hpp
        include/accelerator.hpp
        include/opencl_kernels.hpp
)

target_link_libraries(cuda_app Qt5::Widgets)
target_link_libraries(cuda_app ${CUDA_LIBRARIES})
target_link_libraries(cuda_app ${OpenCL_LIBRARY})
target_link_libraries(cuda_app ${OpenCV_LIBS})



