find_package(GLUT REQUIRED)
find_package(Threads REQUIRED)
set(OpenGL_GL_PREFERENCE LEGACY)
find_package(OpenGL REQUIRED)

set(eqviolib_visualisation_SOURCE_FILES
    src/Plotter.cpp
)

set(eqviolib_visualisation_HEADER_FILES
    include/Plotter.h
    include/PlotDataTypes.h
)

add_library(eqviolib_visualisation
    ${eqviolib_visualisation_HEADER_FILES}
    ${eqviolib_visualisation_SOURCE_FILES}
)

target_include_directories(eqviolib_visualisation
    PRIVATE include
    PRIVATE ${EIGEN_INCLUDE_DIRS}
    PRIVATE ${GLUT_INCLUDE_DIRS}
    PRIVATE ${OPENGL_INCLUDE_DIR}
)

target_link_libraries(eqviolib_visualisation
    ${GLUT_LIBRARY}
    ${OPENGL_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)
