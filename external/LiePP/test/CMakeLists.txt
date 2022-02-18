
# Get gtest as an external project
if (NOT TARGET gtest_main)
    include(PrepareGTest.cmake)
endif()
# This creates the linking target gtest_main

set(TESTS_LIST
    test_groups
)

foreach(TEST_NAME ${TESTS_LIST})
    add_executable(${TEST_NAME} "${TEST_NAME}.cpp")
    target_include_directories(${TEST_NAME} PUBLIC
        include
        test
        ${EIGEN_INCLUDE_DIRS}
    )
    target_link_libraries(${TEST_NAME}
        gtest_main
        liepp
    )
    add_test("${TEST_NAME}" ${TEST_NAME})
endforeach()