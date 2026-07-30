cmake_minimum_required(VERSION 3.16)
project(xsmart_lane_geometry_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(xsmart_lane_geometry SHARED
    src/lane_geometry_backend.cpp
)
target_include_directories(xsmart_lane_geometry PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(xsmart_lane_geometry PROPERTIES
    OUTPUT_NAME xsmart_lane_geometry
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(xsmart_lane_geometry PRIVATE
        -O3
        -Wall
        -Wextra
        -Wpedantic
    )
endif()
