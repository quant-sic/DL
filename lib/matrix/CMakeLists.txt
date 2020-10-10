cuda_add_library( matrix_src
    
    src/matrix.cpp
    src/matrix.h

    )

#define include path    
target_include_directories(matrix_src PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
