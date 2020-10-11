/*
  test the network step by step
*/

#define _DEBUG_NEURAL_OPT_ 1

#include "neural_network.h"
#include "test_network.h"

// main fucntion to run step by step
int main(int argc, char** argv) {
  // standard parameters
  double learn_rate = 0.01;
  int    size_batch = 5;
  int    n_batches  = 1;
  bool   flag_host = true;
  
  // neruon parameters
  int neurons_in = 10;
  int neurons_mid = 10;
  int neurons_out = 10;
  
  // creating the input samples
  std::vector <matrix> v_m_inp = create_sample_one_hot(neurons_in,  n_batches, size_batch);
  std::vector <matrix> v_m_out = create_sample_one_hot(neurons_out, n_batches, size_batch);
  
  // create neural network
  neural_network nn(learn_rate, flag_host);
  
  // add the layers
  nn.add_layer(new linear ("lini_0", neurons_in, neurons_mid));
  nn.add_layer(new sigmoid("siggi_0"));
  nn.add_layer(new linear ("lini_1", neurons_mid, neurons_out));
  nn.add_layer(new sigmoid("siggi_1"));

  // adding the cost
  nn.set_cost(new rms_cost());
  
  // move matrices to host and device
  if (!flag_host)
    for (int idx = 0; idx < v_m_inp.size(); idx++) {
      v_m_inp[idx].copy_host_to_device();
      v_m_out[idx].copy_host_to_device();
    }
  
  // propagating the network
  matrix y;
  for (int idx_b = 0; idx_b < n_batches; idx_b++) {
    y = nn.prop_forward(v_m_inp[idx_b]);
    nn.prop_backward(y, v_m_out[idx_b]);
  }
}
