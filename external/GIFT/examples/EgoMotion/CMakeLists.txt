# find_package(GIFT)
find_package(OpenCV REQUIRED)

add_executable(EgoMotion main.cpp)

target_include_directories(EgoMotion PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(EgoMotion GIFT ${OpenCV_LIBS})
