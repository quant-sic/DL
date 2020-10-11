cuda_add_library(lib_binding SHARED
    binding.cu
    )

#define include path    
target_link_libraries(lib_binding PRIVATE matrix_src costs_src lib_layers_src lib_neural_network_src lib_neural_network_test lib_mnist_reader)
set_target_properties(lib_binding PROPERTIES PREFIX "")


cuda_add_library(lib_comp_utils

    common_utils.h
    device_utils.h
    host_utils.h
    )

#define include path    
target_include_directories(lib_comp_utils PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")


cuda_add_library(lib_mnist_reader

    mnist_reader.cu
    mnist_reader.h
    )

#define include path    
target_include_directories(lib_mnist_reader PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(lib_mnist_reader PUBLIC matrix_src)
