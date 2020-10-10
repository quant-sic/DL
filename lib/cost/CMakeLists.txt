cuda_add_library(costs_src
    
    costfunctions.cu
    costfunctions.h
    costs.cpp
    costs.h

    )


#define include path    
target_include_directories(costs_src PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(costs_src PRIVATE matrix_src misc_src activations_src matrix_operations_src)
