cuda_add_library( misc_src
    
    common.h
    common.c
    global.h
    global.cu
    timer.cu
    timer.h
    )

#define include path    
target_include_directories(misc_src PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
