project(common)

set(COMMON_SOURCES
        matrix_converter.cpp
        matrix_converter.h)

add_library(common ${COMMON_SOURCES})
target_include_directories(common PUBLIC .)
target_link_libraries(common MatrixMarketReader)