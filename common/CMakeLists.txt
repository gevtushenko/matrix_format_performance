project(common)

set(COMMON_SOURCES
        matrix_converter.cpp
        matrix_converter.h measurement_class.cpp measurement_class.h scoo_matrix_class.cpp scoo_matrix_class.h)

add_library(common ${COMMON_SOURCES})
target_include_directories(common PUBLIC .)
target_link_libraries(common MatrixMarketReader)