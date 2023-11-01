cmake_minimum_required(VERSION 3.5)

project(sam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_53;code=sm_53)


find_package(OpenCV 4 REQUIRED )

link_directories(
    "/usr/local/cuda/lib64"
    "/usr/local/TensorRT-8.5.3.1/lib"
)

include_directories(
    "/usr/local/TensorRT-8.5.3.1/include"
)


include_directories(
    ${OpenCV_INCLUDE_DIRS}/include
    ${CMAKE_SOURCE_DIR}/include/
    ${CUDA_INCLUDE_DIRS}
)

file(GLOB SOURCE
    ./src/*.cpp
)

add_executable(sam ${SOURCE} main.cpp)
target_link_libraries(sam ${OpenCV_LIBS} cudart nvinfer cudart nvonnxparser nvinfer_plugin)
