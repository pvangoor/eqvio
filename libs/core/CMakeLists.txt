set(eqviolib_core_HEADER_FILES
    include/common.h
    include/LieYaml.h
)

add_library(eqviolib_core INTERFACE)

target_include_directories(eqviolib_core
    INTERFACE include
    INTERFACE ${LiePP_INCLUDE_DIRS}
)

target_link_libraries(eqviolib_core
    INTERFACE yaml-cpp
)
