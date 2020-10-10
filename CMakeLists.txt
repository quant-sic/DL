cmake_minimum_required(VERSION 2.8)

project(DL
    DESCRIPTION "DL Algorithms Suite"
    LANGUAGES CXX C
    )


find_package(PythonLibs 2.7 REQUIRED)
find_package(BLAS REQUIRED)
message(${BLAS_LIBRARIES})
find_package(CUDA)

# # Pass options to NVCC
set(CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS};
  # 03 #-arch sm_30
  -Xptxas -O3
  -Xcudafe "--diag_suppress=unsigned_compare_with_zero"
  ) 

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ./lib/pybind11/include
  )

cmake_policy(SET CMP0023 OLD)

add_subdirectory(lib/pybind11)
add_subdirectory(lib/misc)

add_subdirectory(lib/activations)
add_subdirectory(lib/matrix)
add_subdirectory(lib/layers)
add_subdirectory(lib/cost)
add_subdirectory(lib/matrix_operations)

add_subdirectory(test_scripts)

