/*
  TEST THE NETWORK HEADER
  
  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    :
  TO-DO   :
  CAUTION : 
*/

#ifndef _TEST_NETWORK_H_
#define _TEST_NETWORK_H_

// c++ standard headers 
#include <vector>
#include <string>
#include <iostream>
#include <sstream>

// own c++ headers
#include "../neural_network/matrix.h"
#include "../neural_network/neural_network.h"

//_______________________________________________________________________________________________
// functions contained in this module
neural_network create_neural_network(const std::vector <std::string>& v_layers, const std::vector <int>& v_neurons, const std::string& name_cost, bool flag_host, double learning_rate);

// creating input / output for neural net
std::vector <matrix> create_sample_one_hot(int neurons, int n_batches, int size_batch);

// running a neural network
bool run_network_test(neural_network& nn, std::vector <matrix>& v_m_inp, std::vector <matrix>& v_m_out, int epochs);
std::pair <double, double> time_network_test(neural_network& nn, std::vector <matrix>& v_m_inp, std::vector <matrix>& v_m_out);

// accuracy computation
int    get_predicted_number(const matrix& mat, const int& idx_bat);
double compute_accuracy    (const std::vector <matrix>& v_predicts, const std::vector <matrix>& v_targets);
double compute_accuracy    (const std::vector <matrix>& v_predicts, const std::vector <matrix>& v_targets, int n_targets);

// helper functions to compute number of flop per layer operation
size_t flop_neural_network_forward(std::vector <int> v_sizes, std::vector <std::string> v_names, size_t size_batch);
size_t flop_neural_network_backward(std::string cost, std::vector <int> v_sizes, std::vector <std::string> v_names, size_t size_batch);
size_t flop_softmax_forward (size_t size_batch, size_t neurons_out);
size_t flop_softmax_backward(size_t size_batch, size_t neurons_out);
size_t flop_sigmoid_forward (size_t size_batch, size_t neurons_out);
size_t flop_sigmoid_backward(size_t size_batch, size_t neurons_out);
size_t flop_relu_forward    (size_t size_batch, size_t neurons_out);
size_t flop_relu_backward   (size_t size_batch, size_t neurons_out);
size_t flop_linear_forward  (size_t neurons_in, size_t neurons_out, size_t size_batch);
size_t flop_linear_backward (size_t neurons_in, size_t neurons_out, size_t size_batch);
size_t flop_mat_mul         (size_t rows_lhs, size_t cols_lhs, size_t cols_rhs);
size_t flop_dcce            (size_t size_batch, size_t neurons_out);
size_t flop_drms            (size_t size_batch, size_t neurons_out);
size_t flop_dcce_soft       (size_t size_batch, size_t neurons_out);

// creating combinations with specific size from vector
template <typename T> std::vector <std::vector <T>> create_combinations(const std::vector <T>& v_inp, const size_t& size);
template <typename T> void recursion_combi(std::vector <std::vector<T>>& v_res, const std::vector <T>& v_inp, std::vector <T>& v_dummy, const size_t size, size_t calls);

//_______________________________________________________________________________________________
// does the same as recusrion combi, but without the unnecessary parameters
template <typename T>
std::vector <std::vector <T>> create_combinations(const std::vector <T>& v_inp,
						  const size_t&          size)
{
  // calling recusrion combination in a predifned way
  std::vector <T> v_dummy;
  std::vector <std::vector <T>> vv_result;
  recursion_combi(vv_result, v_inp, v_dummy, size, 0);
  return vv_result;
}

//_______________________________________________________________________________________________
// creating all possible combinations  with specific length from a vector of inputs
// neither fast nor safe -> always a good combination !!!
template <typename T>
void recursion_combi(std::vector <std::vector<T>>& vv_res,
		     const std::vector <T>&        v_inp,
		     std::vector <T>&              v_dummy,
		     const size_t                  size,
		     size_t                        calls)
{
  // add result if depth is reached
  if (calls == size) {
    vv_res.push_back(v_dummy);
    return;
  } else {
    // safety check beginning
    if (calls == 0)
      v_dummy.clear();
    
    // loop over possibilities
    for (int i = 0; i < v_inp.size(); i++) {
      v_dummy.push_back(v_inp[i]);
      recursion_combi(vv_res, v_inp, v_dummy, size, calls+1);
      v_dummy.pop_back();
    }
  }
}

#endif // _TEST_NETWORK_H_
