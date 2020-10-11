cuda_add_library(lib_neural_network_src
    
    src/neural_network.cu
    src/neural_network.h
    )

#define include path    
target_include_directories(lib_neural_network_src PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(lib_neural_network_src PUBLIC costs_src lib_neural_network_test lib_layers_src )


cuda_add_library(lib_neural_network_test
    
    test/test_network.h
    test/test_network.cu

    )

#define include path    
target_include_directories(lib_neural_network_test PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/test")
target_link_libraries(lib_neural_network_test PUBLIC matrix_src misc_src lib_neural_network_src )


cuda_add_library(lib_neural_network_run
    
    run/run_network.h
    run/run_network.cu

    )

#define include path    
target_include_directories(lib_neural_network_run PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/run")
