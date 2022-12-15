
# The order of modules is important!
if (EQVIO_BUILD_VISUALISATION)
set(module visualisation)

set(EQVIO_MODULE_INCLUDE_DIRS "")
set(EQVIO_MODULE_LIBS "")

add_subdirectory(${module})
list(APPEND EQVIO_MODULE_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/libs/${module}/include")
list(APPEND EQVIO_MODULE_LIBS "eqviolib_${module}")

set(EQVIO_MODULE_INCLUDE_DIRS ${EQVIO_MODULE_INCLUDE_DIRS} PARENT_SCOPE)
set(EQVIO_MODULE_LIBS ${EQVIO_MODULE_LIBS} PARENT_SCOPE)
endif()