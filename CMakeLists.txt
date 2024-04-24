cmake_minimum_required(VERSION 3.20)
project(hybrid_feature_extractor)

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

find_package(ament_cmake REQUIRED)
