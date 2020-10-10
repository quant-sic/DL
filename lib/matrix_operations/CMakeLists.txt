cuda_add_library( matrix_operations_src
    
    src/kernel_utils.cu
    src/kernel_utils.h
    src/matrix_operator_gpu.cu
    src/matrix_operator_gpu.h
    src/matrix_operator.cu
    src/matrix_operator.h

    )

cuda_add_library( test_matrix_operations_src
    
    test/test_matrix_operator.cu
    test/test_matrix_operator.h
    )


#define include path 
target_include_directories(matrix_operations_src PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(matrix_operations_src PRIVATE misc_src)

target_include_directories(test_matrix_operations_src PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/test")
target_link_libraries(test_matrix_operations_src PRIVATE matrix_operations_src misc_src)
