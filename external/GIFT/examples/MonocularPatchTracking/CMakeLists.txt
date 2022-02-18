# find_package(GIFT)
find_package(OpenCV REQUIRED)

add_executable(MonocularPatchTracking main.cpp)

target_include_directories(MonocularPatchTracking PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(MonocularPatchTracking GIFT ${OpenCV_LIBS})
