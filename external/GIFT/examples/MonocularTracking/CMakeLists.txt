# find_package(GIFT)
find_package(OpenCV REQUIRED)

add_executable(MonocularTracking main.cpp)

target_include_directories(MonocularTracking PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(MonocularTracking GIFT ${OpenCV_LIBS})
