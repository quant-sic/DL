# add executables for test scripts
# __________________________________________________________________________________________________________________________________


# time and test scripts for algorithms -> replace these with something more clever

cuda_add_executable(test_activations test_activations.cu)
target_link_libraries(test_activations PRIVATE activations_src lib_costs_src misc_src test_matrix_operations_src)

cuda_add_executable(test_CBlas test_CBlas.c)
target_link_libraries(test_CBlas PRIVATE misc_src test_matrix_operations_src -lcblas)

# cuda_add_executable(run_test_matrix_operator run_test_matrix_operator.cu)
# target_link_libraries(run_test_matrix_operator PRIVATE misc_src test_matrix_operations_src matrix_operations_src)

cuda_add_executable(test_costs test_costs.cu)
target_link_libraries(test_costs PRIVATE lib_costs_src misc_src activations_src matrix_operations_src test_matrix_operations_src)

cuda_add_executable(test_ew test_ew.cu)
target_link_libraries(test_ew PRIVATE lib_costs_src misc_src activations_src matrix_operations_src test_matrix_operations_src)

cuda_add_executable(test_ew2d test_ew2d.cu)
target_link_libraries(test_ew2d PRIVATE misc_src matrix_operations_src test_matrix_operations_src)

cuda_add_executable(test_matmul test_matmul.cu)
target_link_libraries(test_matmul PRIVATE misc_src matrix_operations_src test_matrix_operations_src)

# cuda_add_executable(time_ew time_ew.cu)
# target_link_libraries(time_ew PRIVATE lib_costs_src misc_src activations_src matrix_operations_src test_matrix_operations_src)

# cuda_add_executable(time_ew2d time_ew2d.cu)
# target_link_libraries(time_ew2d PRIVATE lib_costs_src misc_src activations_src matrix_operations_src test_matrix_operations_src)

# cuda_add_executable(time_matmul time_matmul.cu)
# target_link_libraries(time_matmul PRIVATE misc_src matrix_operations_src test_matrix_operations_src)

# cuda_add_executable(time_reduction time_reduction.cu)
# target_link_libraries(time_reduction PRIVATE lib_costs_src misc_src activations_src matrix_operations_src test_matrix_operations_src)

# cuda_add_executable(time_softmax time_softmax.cu)
# target_link_libraries(time_softmax PRIVATE lib_costs_src misc_src activations_src)

# __________________________________________________________________________________________________________________________________


# measure network
cuda_add_executable(measure_network measure_network.cu)
target_link_libraries(measure_network PRIVATE 
    misc_src 
    test_matrix_operations_src 
    lib_neural_network_test 
    lib_neural_network_src 
    lib_neural_network_run
    matrix_src 
    lib_mnist_reader  
    )


# test layers
cuda_add_executable(test_layers test_layers.cu)
target_link_libraries(test_layers PRIVATE lib_layers_src lib_layers_test lib_matrix_test)

# test mnist reader
cuda_add_executable(test_mnist_reader test_mnist_reader.cu)
target_link_libraries(test_mnist_reader PRIVATE lib_mnist_reader)

#test network
cuda_add_executable(test_network test_network.cu)
target_link_libraries(test_network PRIVATE lib_neural_network_test)

# test network step by step
cuda_add_executable(test_step_network test_step_network.cu)
target_link_libraries(test_step_network PRIVATE lib_neural_network_test lib_neural_network_src)

#time network
cuda_add_executable(time_network time_network.cu)
target_link_libraries(time_network PRIVATE lib_neural_network_test lib_neural_network_src lib_neural_network_run lib_mnist_reader)



#time network
cuda_add_executable(test_functor_cuda test_functor_cuda.cu)
target_link_libraries(test_functor_cuda PRIVATE misc_src test_matrix_operations_src)
