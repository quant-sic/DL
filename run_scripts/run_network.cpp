/*
  AUTHOR  : FABIAN DECHENT / JANNIS SCHÜRMANN 
  DATE    : 
  TO-DO   :
  CAUTION : 
*/

#include "run_network.h"

//_____________________________________________________________________________________________
// read in function for parameters
std::tuple<bool, bool, std::vector<int>, std::vector<std::string>, std::string, double, int, int> network_read_in(void)
{
  std::cout << "____________________________________________________\n";
  std::cout << "NEURAL NETWORK\n";
  std::cout << "\n";

  // read in of letters / digits
  bool flag_digit;
  std::string flag_digits_letters;
  while(true) {
    std::cout << "Should the Network be trained on letters or digits? (letters/digits)   : ";
    std::cin >> flag_digits_letters;
    if (flag_digits_letters == "letters") {
      flag_digit = false;
      break;
    } else if (flag_digits_letters == "digits") {
      flag_digit = true;
      break;
    } else
      std::cout << "The choosen option is not available. Try again." << std::endl;
  }
  
  // read in the number of hidden layers
  int n_layers;
  while(true) {
    std::cout << "How many hidden layers should the network have? (int)                  : ";
    std::cin >> n_layers;
    if (n_layers >= 0)
      break;
    else
      std::cout << "The Network can't have a negative number of hidden layers." << std::endl;  
  }
  
  // reading in the information about the layers
  int neurons;
  std::vector <int> v_neurons = {784}; // maybe not hardcode the image size
  for (int i = 0; i < n_layers; i++) {
    while(true) {
      std::cout << "How many neurons should hidden layer " << i << " have? (int)                     : ";
      std::cin >> neurons;
      if (neurons > 0) {
	v_neurons.push_back(neurons);
	break;
      } else
	std::cout << "The number of neurons should be larger than 0." << std::endl;
    }
  }
  
  // add one because magic
  n_layers++;
  
  // reading in the activations
  std::string activation;
  std::vector <std::string> v_layers;
  for (int i = 0; i < n_layers; i++) {
    while(true) {
      std::cout << "What shoule be the activation " << i << "? (sigmoid/relu/softmax)                : ";
      std::cin >> activation;
      if (activation == "relu" or activation == "sigmoid" or activation == "softmax") {
	v_layers.push_back(activation);
	break;
      } else {
	std::cout << "The choosen activation option is not available." << std::endl;
      }
    }
  }
  
  // add the size of the output to the neurons vector
  int n_target_neurons;
  if (flag_digits_letters == "letters")
    n_target_neurons = 26;
  else if (flag_digits_letters == "digits")
    n_target_neurons = 10;
  v_neurons.push_back(n_target_neurons);
  
  // read in cost function
  std::string cost_function;
  while(true) {
    std::cout << "What cost function should be used? (cce/rms/cce_soft)                  : ";
    std::cin >> cost_function;
    if (cost_function == "cce" or cost_function == "rms" or cost_function == "cce_soft")
      break;
    else
      std::cout << "The choosen cost function option is not available." << std::endl;
  }
  
  // reading in additional information
  double learning_rate;
  while(true) {
    std::cout << "What should the learning_rate be? (double (0,1))                       : ";
    std::cin >> learning_rate;
    if (learning_rate >= 0.0 and learning_rate <= 1.0)
      break;
    else
      std::cout << "The learning rate should be 0.0 <= learn_rate <= 1.0." << std::endl;
  }
  
  // read in the number of epochs
  int epochs;
  while(true) {
    std::cout << "How many epochs should the network be trained? (int)                   : ";
    std::cin >> epochs;
    if (epochs > 0)
      break;
    else
      std::cout << "The number of epochs should be larger than 0." << std::endl;
  }
  
  // host or device option
  bool flag_host = false;
  std::string flag_host_device = "device";
  while(true) {
    std::cout << "Should the calculation take place on the host or device? (host/device) : ";
    std::cin >> flag_host_device; 
    if (flag_host_device == "host") {
      flag_host = true;
      break;
    } else if (flag_host_device == "device") {
      flag_host = false;
      break;
    } else
      std::cout << "The choosen host/device option not available." << std::endl;
  }
  
  // read in size batch
  int size_batch;
  while(true) {
    std::cout << "How big should the batch size be? (int)                                : ";
    std::cin >> size_batch;
    if (size_batch >= 0)
      break;
    else
      std::cout << "The batch size needs to be larger than 0." << std::endl;
  }

  return std::make_tuple(flag_digit, flag_host, v_neurons, v_layers, cost_function, learning_rate, epochs, size_batch);
}
