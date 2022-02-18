# find_package(GIFT)
find_package(OpenCV REQUIRED)

add_executable(VisualOdometryTracking main.cpp)

target_include_directories(VisualOdometryTracking PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(VisualOdometryTracking GIFT ${OpenCV_LIBS})
