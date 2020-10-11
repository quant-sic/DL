cuda_add_library(activations_src
    
    src/activations.cu
    src/activations.h
    src/relu.cpp
    src/relu.h
    src/sigmoid.cpp
    src/sigmoid.h
    src/softmax.cpp
    src/softmax.h
    src/tanh.h

    )


#define include path    
target_include_directories(activations_src PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(activations_src PRIVATE layers_src misc_src matrix_operations_src)
