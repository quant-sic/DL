/*
  TEST SCRIPT TO TEST THE NEURAL NETWORK
  
  // TEST DIFFERENT LAYER COMBINATIONS WITH DIFFERENT COST FUNCTIONS
  
  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 26.08.2020
  TO-DO   : 
  CAUTION : 1. v_neurons.size() should be always n_layers+1
*/

#include "test_network.h"

// main fucntion for running the tests
int main(int argc, char** argv)
{
  srand(time(NULL));
  // neural network parameters
  size_t n_layers             = 2;
  int    n_batches            = 10;
  int    size_batch           = 100;
  int    epochs               = 100;
  double learning_rate        = 0.1;
  std::vector <int> v_neurons = {100, 50, 10};
  
  // different costs
  std::vector <std::string> v_costs  = {"cce",  "cce_soft", "rms"};
  std::vector <std::string> v_layers = {"relu", "sigmoid",  "softmax"};
  
  // create different network combinations
  std::vector <std::vector <std::string>> vv_layers = create_combinations(v_layers, n_layers);
  
  // creating input and output
  std::vector <matrix> v_m_inp = create_sample_one_hot(v_neurons.front(), n_batches, size_batch);
  std::vector <matrix> v_m_out = create_sample_one_hot(v_neurons.back(),  n_batches, size_batch);

  // testing all possible network "combinations"
  bool flag_host = true;
  int failed = 0, correct = 0;
  for (int host = 0; host < 2; host++) {
    if (host)
      flag_host = false;
    for (int cost = 0; cost < v_costs.size(); cost++) {
      std::string cost_name = v_costs[cost];
      for (int nn_type = 0; nn_type < vv_layers.size(); nn_type++) {
	// network creation
	neural_network nn = create_neural_network(vv_layers[nn_type], v_neurons, cost_name, flag_host, learning_rate);
	// output for each test
	if (!run_network_test(nn, v_m_inp, v_m_out, epochs))
	  failed++;
	else
	  correct++;
      }
    }
  }

  // output or correct and failed things
  std::cout << "______________________________" << std::endl;
  std::cout << "RUN NETWORK TEST : " << std::endl;
  std::cout << "n tests : " << correct + failed << std::endl;
  std::cout << "correct : " << correct << std::endl;
  std::cout << "failed  : " << failed << std::endl;
}
