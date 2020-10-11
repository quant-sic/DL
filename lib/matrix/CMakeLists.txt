cuda_add_library( matrix_src
    
    src/matrix.cpp
    src/matrix.h

    )

#define include path    
target_include_directories(matrix_src PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(matrix_src PUBLIC misc_src ${PYTHON_LIBRARIES})


cuda_add_library( lib_matrix_test
    
    test/test_matrices_cpp.cpp
    test/test_matrices_cpp.h

    )

#define include path    
target_include_directories(lib_matrix_test PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/test")
target_link_libraries(lib_matrix_test PRIVATE test_matrix_operations_src misc_src matrix_src)
