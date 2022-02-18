
# Get gtest as an external project
if (NOT TARGET gtest_main)
    include(PrepareGTest.cmake)
endif()
# This creates the linking target gtest_main

# Link the tests with the test data
add_compile_definitions(TEST_DATA_DIR="${TEST_DATA_DIR}/")

set(TESTS_LIST
    test_Camera
    test_EgoMotion
    test_ImagePyramid
    test_OptimiseParameters
    test_PatchFeatureTracker
    test_KeyPointFeatureTracker
    test_Calibration
    test_RANSAC
)

foreach(TEST_NAME ${TESTS_LIST})
    add_executable(${TEST_NAME} "${TEST_NAME}.cpp")
    target_include_directories(${TEST_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${TEST_NAME}
        gtest_main
        GIFT
    )
    add_test("${TEST_NAME}" ${TEST_NAME})
endforeach()