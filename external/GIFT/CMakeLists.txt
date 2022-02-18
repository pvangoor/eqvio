cmake_minimum_required(VERSION 3.0)
project(GIFT
VERSION 0.2.0)
set(CMAKE_CXX_STANDARD 20)

# Use O2 level optimisation by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    message("Using the default build type: ${CMAKE_BUILD_TYPE}")
endif(NOT CMAKE_BUILD_TYPE)

option( USE_MARCH_NATIVE "Use the flag -march=native" OFF)
if (USE_MARCH_NATIVE AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    else()
        message("march=native was requested but is not supported.")
    endif()
endif()

# Select floating point precision
option(GIFT_USE_SINGLE_FLOAT "Toggle ON/OFF to use double/single precision floats." OFF)

add_subdirectory(GIFT)

# uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

option( BUILD_EXAMPLES "Build Examples" ON)
if(BUILD_EXAMPLES)
    message("Building the examples...")
    add_subdirectory(examples)
endif()

option( BUILD_TESTS "Build Tests" ON)
set(TEST_DATA_DIR "test/data/")
if(BUILD_TESTS)
    enable_testing()
    message("Building the tests...")
    get_filename_component(TEST_DATA_DIR ${TEST_DATA_DIR} ABSOLUTE)
    message("Expecting test data in ${TEST_DATA_DIR}/")
    add_subdirectory(test)
endif()
