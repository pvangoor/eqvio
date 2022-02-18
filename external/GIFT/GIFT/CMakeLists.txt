# Set up output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")

include_directories("${PROJECT_BINARY_DIR}")

# Include libraries from outside project
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

set(ADDITIONAL_INCLUDE_DIRS
${EIGEN3_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIRS}/unsupported
${OpenCV_INCLUDE_DIRS}
)


# Find project files
####################

set(GIFT_SOURCE_FILES
    src/camera/camera.cpp
    src/camera/DoubleSphereCamera.cpp
    src/camera/EquidistantCamera.cpp
    src/camera/PinholeCamera.cpp
    src/camera/StandardCamera.cpp
    src/PointFeatureTracker.cpp
    src/EgoMotion.cpp
    src/Feature.cpp
    src/ImagePyramid.cpp
    src/OptimiseParameters.cpp
    src/ParameterGroup.cpp
    src/GIFeatureTracker.cpp
    src/KeyPointFeatureTracker.cpp
    src/Visualisation.cpp
    src/Calibration.cpp
    src/RANSAC.cpp
)

set(GIFT_HEADER_FILES
    include/GIFT/camera/camera.h
    include/GIFT/camera/DoubleSphereCamera.h
    include/GIFT/camera/EquidistantCamera.h
    include/GIFT/camera/GICamera.h
    include/GIFT/camera/PinholeCamera.h
    include/GIFT/camera/StandardCamera.h
    include/GIFT/Camera.h
    include/GIFT/PointFeatureTracker.h
    include/GIFT/Feature.h
    include/GIFT/EgoMotion.h
    include/GIFT/ImagePyramid.h
    include/GIFT/OptimiseParameters.h
    include/GIFT/ParameterGroup.h
    include/GIFT/GIFeatureTracker.h
    include/GIFT/PatchFeatureTracker.h
    include/GIFT/KeyPointFeatureTracker.h
    include/GIFT/Visualisation.h
    include/GIFT/Calibration.h
    include/GIFT/RANSAC.h
)

# Add files from within the project
add_library(GIFT STATIC
    ${GIFT_SOURCE_FILES}
    ${GIFT_HEADER_FILES}
)

# Set the floating point type
if (${GIFT_USE_SINGLE_FLOAT})
    target_compile_definitions(GIFT PUBLIC USE_SINGLE_FLOAT)
endif()

# Build local library
#####################
target_include_directories(GIFT PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  
    $<INSTALL_INTERFACE:include>
    ${ADDITIONAL_INCLUDE_DIRS}
)

target_link_libraries(GIFT
    PUBLIC ${OpenCV_LIBS}
    PUBLIC ${YAML_CPP_LIBRARIES}
    # Eigen3::Eigen
)

# INSTALLATION
##############

set(config_install_dir "lib/cmake/${PROJECT_NAME}")
set(include_install_dir "include")
set(project_config "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake")
set(targets_export_name "${PROJECT_NAME}Targets")

include(CMakePackageConfigHelpers)

configure_package_config_file(
    "modules/Config.cmake.in"
    "${project_config}"
    INSTALL_DESTINATION "${config_install_dir}")


# Export '<PROJECT-NAME>Targets.cmake' to build dir (to find package in build dir without install)
export(TARGETS GIFT FILE "${CMAKE_CURRENT_BINARY_DIR}/${targets_export_name}.cmake")

# Targets:
install(TARGETS GIFT
        EXPORT "${targets_export_name}"
        LIBRARY DESTINATION "lib"
        ARCHIVE DESTINATION "lib"
        RUNTIME DESTINATION "bin"
        INCLUDES DESTINATION "${include_install_dir}")

# Config
install(FILES "${project_config}"
        DESTINATION "${config_install_dir}")

# Config
install(EXPORT "${targets_export_name}"
        # NAMESPACE "${namespace}"
        DESTINATION "${config_install_dir}")
# message("The namespace is ${namespace}")

# Headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/GIFT
        DESTINATION ${include_install_dir} 
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
