cuda_add_library( matrix_operations_src
    
    # src/matrix_operator_gpu.cu
    # src/matrix_operator_gpu.h
    src/reduce.cu
    src/reduce.h
    src/mat_mul.cu
    src/mat_mul.h
    src/pw_comp.h
    src/pw2d_comp.h
    src/common_utils.h

    )

cuda_add_library( test_matrix_operations_src
    
    test/test_matrix_operator.cu
    test/test_matrix_operator.h
    )


#define include path 
target_include_directories(matrix_operations_src PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(matrix_operations_src PRIVATE misc_src -lcublas)

target_include_directories(test_matrix_operations_src PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/test")
target_link_libraries(test_matrix_operations_src PRIVATE matrix_operations_src misc_src -lcublas)
