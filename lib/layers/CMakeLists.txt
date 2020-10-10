cuda_add_library( layers_src
    
    src/dense.cpp
    src/dense.h
    src/layers.cpp
    src/layers.h
    src/linear.cpp
    src/linear.h
    src/linear_propagation.cu
    src/linear_propagation.h
    )

#define include path    
target_include_directories(layers_src PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(layers_src PRIVATE matrix_src misc_src matrix_operations_src test_matrix_operations_src activations_src)
