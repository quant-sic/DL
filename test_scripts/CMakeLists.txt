cuda_add_executable(test_activations test_activations.cu)

target_link_libraries(test_activations PRIVATE activations_src costs_src misc_src test_matrix_operations_src -lcblas -lcublas -lcurand ${PYTHON_LIBRARIES})