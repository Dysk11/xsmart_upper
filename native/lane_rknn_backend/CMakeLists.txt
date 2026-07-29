cmake_minimum_required(VERSION 3.16)
project(xsmart_lane_rknn_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_path(RKNN_INCLUDE_DIR rknn_api.h
    HINTS
        $ENV{RKNN_INCLUDE_DIR}
        /usr/include
        /usr/local/include
    PATH_SUFFIXES rknn_api
)
find_library(RKNN_LIBRARY
    NAMES rknnrt
    HINTS
        $ENV{RKNN_LIBRARY_DIR}
        /usr/lib
        /usr/local/lib
        /usr/lib/aarch64-linux-gnu
)

if(NOT RKNN_INCLUDE_DIR OR NOT RKNN_LIBRARY)
    message(FATAL_ERROR "RKNN C API headers/library not found; set RKNN_INCLUDE_DIR and RKNN_LIBRARY_DIR")
endif()

find_path(RGA_INCLUDE_DIR
    NAMES im2d.h
    HINTS
        $ENV{RGA_INCLUDE_DIR}
        /usr/include
        /usr/local/include
    PATH_SUFFIXES rga
)
find_library(RGA_LIBRARY
    NAMES rga
    HINTS
        $ENV{RGA_LIBRARY_DIR}
        /usr/lib
        /usr/local/lib
        /usr/lib/aarch64-linux-gnu
)

add_executable(lane_rknn_backend
    src/main.cpp
)
target_include_directories(lane_rknn_backend PRIVATE
    ${RKNN_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(lane_rknn_backend PRIVATE
    ${RKNN_LIBRARY}
    pthread
    rt
)

if(RGA_INCLUDE_DIR AND RGA_LIBRARY)
    target_include_directories(lane_rknn_backend PRIVATE ${RGA_INCLUDE_DIR})
    target_link_libraries(lane_rknn_backend PRIVATE ${RGA_LIBRARY})
    target_compile_definitions(lane_rknn_backend PRIVATE XSMART_HAVE_RGA=1)
else()
    message(WARNING "librga not found; native backend will retain the direct/CPU preprocessing path")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(lane_rknn_backend PRIVATE
        -O3
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

