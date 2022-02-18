
# Get gtest as an external project
if (NOT TARGET gtest_main)
    include(PrepareGTest.cmake)
endif()
# This creates the linking target gtest_main

# Add some testing utility functions
add_library(testing_utilities_lib
    testing_utilities.h
    testing_utilities.cpp
)
target_include_directories(testing_utilities_lib PUBLIC
    include
    test
    ${EIGEN_INCLUDE_DIRS}
)
target_link_libraries(testing_utilities_lib
    eqvio_lib
)

set(TESTS_LIST
    test_settings
    test_VIOGroup
    test_VIOGroupActions
    test_VIOLift
    test_CoordinateCharts
    test_EqFMatrices
)

set(TEST_REPS 25)
set(NEAR_ZERO 1e-12)

set(EQVIO_DEFAULT_CONFIG_FILE "../EQVIO_config_template.yaml")
get_filename_component(EQVIO_DEFAULT_CONFIG_FILE ${EQVIO_DEFAULT_CONFIG_FILE} ABSOLUTE)
add_compile_definitions(EQVIO_DEFAULT_CONFIG_FILE="${EQVIO_DEFAULT_CONFIG_FILE}")

foreach(TEST_NAME ${TESTS_LIST})
    add_executable(${TEST_NAME} "${TEST_NAME}.cpp")
    target_include_directories(${TEST_NAME} PUBLIC
        include
        test
        ${EIGEN_INCLUDE_DIRS}
    )
    target_link_libraries(${TEST_NAME}
        gtest_main
        eqvio_lib
        testing_utilities_lib
    )
    target_compile_definitions(${TEST_NAME}
    PUBLIC TEST_REPS=${TEST_REPS}
    PUBLIC NEAR_ZERO=${NEAR_ZERO}
    )
    add_test("${TEST_NAME}" ${TEST_NAME})
endforeach()