cmake_minimum_required(VERSION 3.16)
project(xsmart_object_rknn_backend LANGUAGES CXX)

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

add_library(xsmart_object_rknn SHARED
    src/object_backend.cpp
)
target_include_directories(xsmart_object_rknn
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${RKNN_INCLUDE_DIR}
)
target_link_libraries(xsmart_object_rknn PRIVATE ${RKNN_LIBRARY} pthread)
set_target_properties(xsmart_object_rknn PROPERTIES
    OUTPUT_NAME xsmart_object_rknn
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(xsmart_object_rknn PRIVATE
        -O3
        -Wall
        -Wextra
        -Wpedantic
    )
endif()
