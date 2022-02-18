
# The order of modules is important!
set(eqviolib_dirs
    core
)
if (EQVIO_BUILD_VISUALISATION)
    list(APPEND eqviolib_dirs visualisation)
endif()

set(EQVIO_MODULE_INCLUDE_DIRS "")
set(EQVIO_MODULE_LIBS "")

foreach(module ${eqviolib_dirs})
    add_subdirectory(${module})
    list(APPEND EQVIO_MODULE_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/libs/${module}/include")
    list(APPEND EQVIO_MODULE_LIBS "eqviolib_${module}")
endforeach()

set(EQVIO_MODULE_INCLUDE_DIRS ${EQVIO_MODULE_INCLUDE_DIRS} PARENT_SCOPE)
set(EQVIO_MODULE_LIBS ${EQVIO_MODULE_LIBS} PARENT_SCOPE)
