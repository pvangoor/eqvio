cmake_minimum_required(VERSION 3.12)
project(eqvio VERSION 0.1)
set(CMAKE_CXX_STANDARD 20)

include(CheckCXXCompilerFlag)


# Options and settings
option( EQVIO_BUILD_TESTS "Build Tests" ON)
option( EQVIO_BUILD_VISUALISATION "Build Visualisation Tool" ON)
option( EQVIO_BUILD_ROSBAG "Build EqVIO with rosbag reading enabled" OFF)
option( EQVIO_SUPPORT_CONCEPTS "Build EqVIO using c++20 concepts" ON)
option( EXTRA_WARNINGS "Enable extra warnings from gcc" ON)

if(EQVIO_SUPPORT_CONCEPTS OR (${CMAKE_CXX_STANDARD} EQUAL 20))
    message("Adding the flag -fconcepts")
    add_compile_options("-fconcepts")
endif()

add_compile_definitions(EQVIO_BUILD_VISUALISATION=$<BOOL:${EQVIO_BUILD_VISUALISATION}>)
add_compile_definitions(EQVIO_BUILD_ROSBAG=$<BOOL:${EQVIO_BUILD_ROSBAG}>)
add_compile_definitions(EQVIO_SUPPORT_CONCEPTS=$<BOOL:${EQVIO_SUPPORT_CONCEPTS}>)

option( EQVIO_USE_MARCH_NATIVE "Use the flag -march=native" OFF)
if (EQVIO_USE_MARCH_NATIVE AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    else()
        message("-march=native was requested but is not supported.")
    endif()
endif()

option( EQVIO_USE_LTO "Use the flag -flto" OFF)
if (EQVIO_USE_LTO AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-flto" COMPILER_SUPPORTS_LTO)
    if(COMPILER_SUPPORTS_LTO)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
    else()
        message("-flto was requested but is not supported.")
    endif()
endif()

option( EQVIO_USE_THREAD_SANTIZER "Use the gcc thread sanitizer option" OFF)
if (EQVIO_USE_THREAD_SANTIZER)
    CHECK_CXX_COMPILER_FLAG("-fsanitize=thread" COMPILER_SUPPORTS_THREAD_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
    if(NOT COMPILER_SUPPORTS_THREAD_SANITIZER)
        message("-fsanitize=thread may not be supported. Trying anyway.")
    endif()
endif()

option( EQVIO_USE_UB_SANTIZER "Use the gcc undefined sanitizer option" OFF)
if (EQVIO_USE_UB_SANTIZER)
    CHECK_CXX_COMPILER_FLAG("-fsanitize=undefined" COMPILER_SUPPORTS_UB_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
    if(NOT COMPILER_SUPPORTS_UB_SANITIZER)
        message("-fsanitize=undefined may not be supported. Trying anyway.")
    endif()
endif()


# Import external dependencies
find_package(Eigen3 3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)


# Add GIFT
find_package(GIFT QUIET)
if (${GIFT_FOUND})
    message("GIFT is being used as an installed package.")
    set(GIFT_INCLUDE_DIRS "GIFT::GIFT")
else()
    message("GIFT is being used as a git submodule.")
    execute_process(COMMAND git submodule update --init --recursive --remote
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/external/GIFT
                    RESULT_VARIABLE GIFT_SUBMOD_RESULT)
    if (${GIFT_SUBMOD_RESULT})
        message(FATAL_ERROR "GIFT was not found as a package and could not be included as a submodule.")
    endif()
    add_subdirectory(external/GIFT)
    set(GIFT_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/external/GIFT/GIFT/include")
endif()

# Add argparse
find_package(argparse QUIET)
if (${argparse_FOUND})
    message("argparse is being used as an installed package.")
    set(argparse_INCLUDE_DIRS "argparse::argparse")
else()
    message("argparse is being used as a git submodule.")
    execute_process(COMMAND git submodule update --init --recursive --remote
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/external/argparse
                    RESULT_VARIABLE argparse_SUBMOD_RESULT)
    if (${argparse_SUBMOD_RESULT})
        message(FATAL_ERROR "argparse was not found as a package and could not be included as a submodule.")
    endif()
    set(argparse_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/external/argparse/include")
endif()

# Add LiePP
message("LiePP is being used as a git submodule.")
execute_process(COMMAND git submodule update --init --recursive --remote
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/external/LiePP
                RESULT_VARIABLE LiePP_SUBMOD_RESULT)
if (${argparse_SUBMOD_RESULT})
    message(FATAL_ERROR "LiePP could not be included as a submodule.")
endif()
set(LiePP_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/external/LiePP/include")

# Add libs
add_subdirectory(libs)

# Add the include files
set(EQVIO_HEADER_FILES
    include/eqvio/VIOState.h
    include/eqvio/IMUVelocity.h
    include/eqvio/VIOGroup.h
    include/eqvio/VIOFilter.h
    include/eqvio/VIOFilterSettings.h
    include/eqvio/csv/CSVLine.h
    include/eqvio/csv/CSVReader.h
    include/eqvio/csv/CSVFile.h
    include/eqvio/VisionMeasurement.h
    include/eqvio/EqFMatrices.h
    include/eqvio/Geometry.h
    include/eqvio/VIOSimulator.h
    include/eqvio/dataserver/dataservers.h
    include/eqvio/dataserver/DatasetReaderBase.h
    include/eqvio/dataserver/ASLDatasetReader.h
    include/eqvio/dataserver/APDatasetReader.h
    include/eqvio/dataserver/UZHFPVDatasetReader.h
    include/eqvio/dataserver/DataServerBase.h
    include/eqvio/dataserver/SimpleDataServer.h
    include/eqvio/dataserver/ThreadedDataServer.h
    include/eqvio/LoopTimer.h
    include/eqvio/aofstream.h
    include/eqvio/VIOWriter.h
    )
set(EQVIO_SOURCE_FILES
    src/VIOState.cpp
    src/VIOGroup.cpp
    src/VIOFilter.cpp
    src/VisionMeasurement.cpp
    src/IMUVelocity.cpp
    src/EqFMatrices.cpp
    src/Geometry.cpp
    src/VIOSimulator.cpp
    src/coordinateSuite/euclid.cpp
    src/coordinateSuite/invdepth.cpp
    src/coordinateSuite/normal.cpp
    src/dataserver/ASLDatasetReader.cpp
    src/dataserver/APDatasetReader.cpp
    src/dataserver/UZHFPVDatasetReader.cpp
    src/dataserver/DataServerBase.cpp
    src/dataserver/SimpleDataServer.cpp
    src/dataserver/ThreadedDataServer.cpp
    src/LoopTimer.cpp
    src/VIOWriter.cpp
)

# Try to add rosbag
if (EQVIO_BUILD_ROSBAG)
    find_package(rosbag REQUIRED)
    find_package(cv_bridge REQUIRED)
    set(CMAKE_CXX_STANDARD 17)

    list(APPEND EQVIO_HEADER_FILES
        include/eqvio/dataserver/RosbagDatasetReader.h
        include/eqvio/dataserver/HiltiDatasetReader.h
    )
    list(APPEND EQVIO_SOURCE_FILES
        src/dataserver/RosbagDatasetReader.cpp
        src/dataserver/HiltiDatasetReader.cpp
    )
    message("Building with rosbag support.")
endif()

# Add visualisation if requested
if (EQVIO_BUILD_VISUALISATION)
    list(APPEND EQVIO_HEADER_FILES
        include/eqvio/VIOVisualiser.h
    )
    list(APPEND EQVIO_SOURCE_FILES
        src/VIOVisualiser.cpp
    )
    message("Building with visualisation support.")
endif()

# Add the EqVIO library

add_library(eqvio_lib
    ${EQVIO_SOURCE_FILES}
    ${EQVIO_HEADER_FILES}
)

target_include_directories(eqvio_lib
    PUBLIC include
    PUBLIC ${EQVIO_MODULE_INCLUDE_DIRS}
    PUBLIC ${OPENCV_INCLUDE_DIRS}
    PUBLIC ${GIFT_INCLUDE_DIRS}
    PUBLIC ${LiePP_INCLUDE_DIRS}
    PUBLIC ${EIGEN_INCLUDE_DIRS}
    PUBLIC ${rosbag_INCLUDE_DIRS}
    PUBLIC ${cv_bridge_INCLUDE_DIRS}
)

target_link_libraries(eqvio_lib
    ${EQVIO_MODULE_LIBS}
    ${OpenCV_LIBS}
    GIFT
    ${rosbag_LIBRARIES}
    ${cv_bridge_LIBRARIES}
)

if (${CMAKE_COMPILER_IS_GNUCC} AND ${CMAKE_CXX_COMPILER_VERSION} LESS 9.0)
    message("GCC version is less than 9. Manually linking std::filesystem.")
    target_link_libraries(eqvio_lib stdc++fs)
endif()

if (${EXTRA_WARNINGS})
    message("Extra warnings are enabled.")
    target_compile_options(eqvio_lib PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Add the tests

if(EQVIO_BUILD_TESTS)
    enable_testing()
    message("Building the tests.")
    add_subdirectory(test)
endif()

# Add the main executable
add_executable(eqvio_opt src/main_opt.cpp)
set(EXECUTABLES_LIST
    eqvio_opt
)

foreach(EXECUTABLE_NAME ${EXECUTABLES_LIST})
    target_include_directories(${EXECUTABLE_NAME}
        PRIVATE include
        PRIVATE ${EIGEN_INCLUDE_DIRS}
        PRIVATE ${OpenCV_INCLUDE_DIRS}
        PRIVATE ${YAML_CPP_INCLUDE_DIR}
        PRIVATE ${LiePP_INCLUDE_DIRS}
        PRIVATE ${GIFT_INCLUDE_DIRS}
        PRIVATE ${argparse_INCLUDE_DIRS}
    )
    target_link_libraries(${EXECUTABLE_NAME}
        eqvio_lib
        ${EQVIO_MODULE_LIBS}
        ${OpenCV_LIBS}
        yaml-cpp
        GIFT
    )
endforeach()


# Add documentation
option( EQVIO_BUILD_DOCS "Use doxygen to build documentation" OFF)
if (EQVIO_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    message("Documentation will be built using doxygen.")
    set(DOXYGEN_USE_MATHJAX true)
    set(DOXYGEN_MATHJAX_EXTENSIONS TeX/AMSmath TeX/AMSsymbols)
    doxygen_add_docs(eqvio_docs
        ${EQVIO_HEADER_FILES}
        ${EQVIO_SOURCE_FILES}
        README.md
        ALL
    )
endif()