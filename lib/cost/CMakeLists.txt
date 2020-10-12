cuda_add_library(lib_costs_src
    
    src/costfunctions.cu
    src/costfunctions.h
    src/costs.cu
    src/costs.h
    src/mse_cost.h
    src/mse_cost.cu
    )


#define include path    
target_include_directories(lib_costs_src PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(lib_costs_src PUBLIC matrix_src misc_src activations_src matrix_operations_src)
