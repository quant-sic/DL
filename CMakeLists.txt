cmake_minimum_required(VERSION 2.8)
find_package(PythonLibs 2.7 REQUIRED)
find_package(BLAS REQUIRED)
message(${BLAS_LIBRARIES})

find_package(CUDA)

# Pass options to NVCC
set(CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS};
  # 03 #-arch sm_30
  -Xptxas -O3
  -Xcudafe "--diag_suppress=unsigned_compare_with_zero"
  ) 

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ./src/pybind11/include
  )

# source misc
set(src_misc
  src/common.h
  src/common.c
  src/global.h
  src/global.cu
  )

# source matrix operations
set(src_matrix_operations
  ${src_misc}
  src/matrix_operations/matrix_operator.h
  src/matrix_operations/matrix_operator.cu
  src/matrix_operations/matrix_operator_gpu.h
  src/matrix_operations/matrix_operator_gpu.cu
  src/matrix_operations/kernel_utils.h
  src/matrix_operations/kernel_utils.cu
  )

# source tensor operations
set(src_tensor_operations
  ${src_matrix_operations}
  src/matrix_operations/tensor.h
  #src/matrix_operations/t_contract_d.h
  #src/matrix_operations/t_contract_d.cu
  src/matrix_operations/t_contract_h.h
  src/matrix_operations/t_contract_h.c
  )

# test matrix operator source
set(src_test_matrix_operations
  ${src_matrix_operations}
  src/matrix_operations/test_matrix_operator.h
  src/matrix_operations/test_matrix_operator.cu
  )

# source for the costs
set(src_cost_functions
  src/cost/costs.h
  src/cost/costs.cpp
  src/cost/costfunctions.h
  src/cost/costfunctions.cu
  )

# source for dense layers
set(src_dense_layers
  ${src_misc}
  src/layers/activations.h
  src/layers/activations.cu
  src/layers/Dense.h
  src/layers/Dense.cu
  )

# source cpp layers
set(src_cpp_layers
  ${src_cost_functions}
  src/neural_network/matrix.h
  src/neural_network/matrix.cpp
  src/layers/layers.h
  src/layers/layers.cpp
  src/layers/linear.h
  src/layers/linear.cpp
  src/layers/relu.h
  src/layers/relu.cpp
  src/layers/sigmoid.h
  src/layers/sigmoid.cpp
  src/layers/softmax.h
  src/layers/softmax.cpp
  src/layers/activations.h
  src/layers/activations.cu
  src/layers/linear_propagation.h
  src/layers/linear_propagation.cu
  )

# source for the neural net
set(src_neural_network
  ${src_matrix_operations}
  ${src_cpp_layers}
  src/neural_network/mnist_reader.h
  src/neural_network/mnist_reader.cpp
  src/test_scripts/test_network.h
  src/test_scripts/test_network.cpp
  src/neural_network/neural_network.h
  src/neural_network/neural_network.cpp
  src/timer.h
  src/timer.cpp
  src/run_scripts/run_network.h
  src/run_scripts/run_network.cpp
  )

# source for testing the neural network
set(src_test_neural_network
  ${src_neural_network}
  )

# source for activations
set(src_activations
  src/cost/costs.h
  src/cost/costfunctions.h
  src/cost/costfunctions.cu
  src/layers/activations.h
  src/layers/activations.cu
  src/layers/tanh.h
  ${src_test_matrix_operations}
  )

# source for the layers test
set(src_test_layers
  ${src_misc}
  ${src_cpp_layers}
  src/matrix_operations/matrix_operator.h
  src/matrix_operations/matrix_operator.cu
  src/matrix_operations/matrix_operator_gpu.h
  src/matrix_operations/matrix_operator_gpu.cu
  src/matrix_operations/kernel_utils.h
  src/matrix_operations/kernel_utils.cu
  src/test_scripts/test_matrices_cpp.h
  src/test_scripts/test_matrices_cpp.cpp
  src/test_scripts/test_layers.h
  src/matrix_operations/test_matrix_operator.h
  src/matrix_operations/test_matrix_operator.cu
  )

set(src_bindin_matrix_nn
  ${src_misc}
  src/neural_network/matrix.h
  src/neural_network/matrix.cpp
  src/neural_network/neural_network.h
  src/neural_network/neural_network.cpp
  src/test_scripts/test_network.h
  src/test_scripts/test_network.cpp
  src/layers/activations.h
  src/layers/activations.cu
  src/cost/costs.h
  src/cost/costs.cpp
  src/cost/costfunctions.h
  src/cost/costfunctions.cu
  src/layers/linear_propagation.h
  src/layers/linear_propagation.cu
  src/matrix_operations/matrix_operator.h
  src/matrix_operations/matrix_operator.cu
  )

#_____________________________________________________________
# python binding
cuda_add_library(binding SHARED src/binding.cu ${src_neural_network})
target_link_libraries(binding ${PYTHON_LIBRARIES} ${CUDA_LIBRARIES} -lcblas -lcublas -lcurand)
set_target_properties(binding PROPERTIES PREFIX "")

#_____________________________________________________________
# test actiavtions and cost
cuda_add_executable(test_activations src/test_scripts/test_activations.cu ${src_activations})
target_link_libraries(test_activations -lcblas -lcublas -lcurand)

cuda_add_executable(test_costs src/test_scripts/test_costs.cu ${src_activations})
target_link_libraries(test_costs -lcblas -lcublas -lcurand)

#_____________________________________________________________
# test matrix / tensor device und host executable

cuda_add_executable(test_CBlas src/test_scripts/test_CBlas.c ${src_test_matrix_operations})
target_link_libraries(test_CBlas -lcblas -lcublas -lcurand)

cuda_add_executable(test_matmul src/test_scripts/test_matmul.cu ${src_test_matrix_operations})
target_link_libraries(test_matmul -lcblas -lcublas -lcurand)

cuda_add_executable(test_ew2d src/test_scripts/test_ew2d.cu ${src_test_matrix_operations})
target_link_libraries(test_ew2d -lcblas -lcublas -lcurand)

cuda_add_executable(test_ew src/test_scripts/test_ew.cu ${src_activations})
target_link_libraries(test_ew -lcblas -lcublas -lcurand)

cuda_add_executable(time_matmul src/test_scripts/time_matmul.cu ${src_test_matrix_operations})
target_link_libraries(time_matmul -lcblas -lcublas -lcurand)

cuda_add_executable(time_ew src/test_scripts/time_ew.cu ${src_activations})
target_link_libraries(time_ew -lcblas -lcublas -lcurand)

cuda_add_executable(time_ew2d src/test_scripts/time_ew2d.cu ${src_activations})
target_link_libraries(time_ew2d -lcblas -lcublas -lcurand)

cuda_add_executable(time_reduction src/test_scripts/time_reduction.cu ${src_activations})
target_link_libraries(time_reduction -lcblas -lcublas -lcurand)

cuda_add_executable(time_softmax src/test_scripts/time_softmax.cu ${src_activations})
target_link_libraries(time_softmax -lcblas -lcublas -lcurand)

#_____________________________________________________________
# test matrix operation executable
cuda_add_executable(run_test_matrix_operator src/test_scripts/run_test_matrix_operator.cu ${src_test_matrix_operations})
target_link_libraries(run_test_matrix_operator -lcblas -lcublas -lcurand)

#_____________________________________________________________
# # neural network executable
cuda_add_executable(run_network src/run_scripts/run_network.cu ${src_neural_network})
target_link_libraries(run_network -lcblas -lcublas -lcurand ${PYTHON_LIBRARIES})

#_____________________________________________________________
# test nerual network executable
cuda_add_executable(test_network src/test_scripts/test_network.cu ${src_test_neural_network})
target_link_libraries(test_network -lcblas -lcublas -lcurand ${PYTHON_LIBRARIES})

#_____________________________________________________________
# time neural network executable
cuda_add_executable(time_network src/test_scripts/time_network.cu ${src_test_neural_network})
target_link_libraries(time_network -lcblas -lcublas -lcurand ${PYTHON_LIBRARIES})

#_____________________________________________________________
# test the layers executable
cuda_add_executable(test_layers src/test_scripts/test_layers.cu ${src_test_layers})
target_link_libraries(test_layers -lcblas -lcublas -lcurand ${PYTHON_LIBRARIES})

#_____________________________________________________________
# test the mnist input
cuda_add_executable(test_mnist src/test_scripts/test_mnist_reader.cu ${src_neural_network})
target_link_libraries(test_mnist -lcblas -lcublas -lcurand ${PYTHON_LIBRARIES})

#_____________________________________________________________
# test the network step by step
cuda_add_executable(test_step_network src/test_scripts/test_step_network.cu ${src_neural_network})
target_link_libraries(test_step_network -lcblas -lcublas -lcurand ${PYTHON_LIBRARIES})

#_____________________________________________________________
# measure the network cost and accuracy
cuda_add_executable(measure_network src/test_scripts/measure_network.cpp ${src_neural_network})
target_link_libraries(measure_network -lcblas -lcublas -lcurand ${PYTHON_LIBRARIES})

