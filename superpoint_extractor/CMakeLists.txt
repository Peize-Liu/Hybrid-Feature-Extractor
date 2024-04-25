cmake_minimum_required(VERSION 3.16)
project(super_point LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(NVCC_FLAGS ${NVCC_FLAGS};-O3;-std=c++17;-gencode arch=compute_87,code=sm_87;-Xcompiler -fPIC;)

set(CMAKE_CXX_STANDARD 17)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(DEBUG "Enable debug symbols" ON)
set(DEBUG_LINK_FLAGS "")

if(DEBUG)
    add_definitions(-DDEBUG)
    set(CMAKE_BUILD_TYPE Debug)
    add_compile_options(-g)

    # include(FetchContent)
    # # Also requires one of: libbfd (gnu binutils), libdwarf, libdw (elfutils)
    # # sudo apt install libbfd-dev libdwarf-dev libdw-dev
    # FetchContent_Declare(backward
    #     GIT_REPOSITORY https://github.com/bombela/backward-cpp
    #     GIT_TAG master  # or a version tag, such as v1.6
    #     SYSTEM          # optional, the Backward include directory will be treated as system directory
    # )
    # FetchContent_MakeAvailable(backward)
    # set(DEBUG_LINK_FLAGS ${DEBUG_LINK_FLAGS} Backward::Interface)

else()
    set(CMAKE_BUILD_TYPE Release)
    add_compile_options(-O3)
endif()

# find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)


# static lib
add_library(super_point STATIC
    src/SuperPoint.cpp
    ../tensorrt/src/logger.cpp
    src/CUDA_NMS.cu)

target_include_directories(super_point PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    /root/workspace/hfe/tensorrt/inc
    inc)

target_link_libraries(super_point 
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${DEBUG_LINK_FLAGS}
    spdlog
    nvinfer
    nvinfer_plugin
    nvonnxparser
    Eigen3::Eigen
      yaml-cpp)

add_executable(super_point_test
    src/test_superpoint.cpp
    )

target_link_libraries(super_point_test
    super_point
    yaml-cpp)
