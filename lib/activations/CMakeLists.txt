cuda_add_library(activations_src
    
    src/activations.cu
    src/activations.h
    src/relu.cu
    src/relu.h
    src/sigmoid.cu
    src/sigmoid.h
    src/softmax.cu
    src/softmax.h
    src/tanh.h

    )


#define include path    
target_include_directories(activations_src PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(activations_src PUBLIC lib_layers_src misc_src matrix_operations_src lib_comp_utils)
