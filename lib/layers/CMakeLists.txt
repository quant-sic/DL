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
target_link_libraries(layers_src PUBLIC matrix_src misc_src matrix_operations_src test_matrix_operations_src activations_src)

cuda_add_library( lib_layers_test
    
    test/test_layers.h
    )

#define include path    
target_include_directories(lib_layers_test PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/test")
target_link_libraries(lib_layers_test PRIVATE layers_src lib_matrix_test)
