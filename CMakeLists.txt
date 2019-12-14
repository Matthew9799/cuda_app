cmake_minimum_required(VERSION 3.10)
project(cuda)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

if(UNIX)
    set(CMAKE_CXX_FLAGS -fPIC)

elseif(WIN32)
    set(OpenCV_DIR "C:\\Program Files\\openCV\\openCV\\opencv\\build")
    set(CMAKE_PREFIX_PATH "C:\\Qt\\5.12.6\\msvc2017_64")
endif(UNIX)

find_package(CUDA QUIET REQUIRED)

find_package( OpenCV REQUIRED )

find_package(Qt5 COMPONENTS Widgets REQUIRED)

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

target_link_libraries(cuda_app Qt5::Widgets)
