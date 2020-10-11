/*
  SCRIPT TO MEASURE THE NETWORK TIMINING AND GFLOPS DEPENDING ON BATCH SIZE

  AUTHOR  : FABIAN DECHENT / JANNIS SCHÜRMANN
  DATE    : 
  TO-DO   : 1. clean up repeated loop creation neural network, cheap trick
  CAUTION : 
*/

// c++ standard headers
#include <iomanip>
#include <iostream>
#include <fstream>

// own c++ headers
#include "test_network.h"
#include "run_network.h"
#include "mnist_reader.h"
#include "neural_network.h"

// main function to run the network test
int main(int argc, char** argv) {
  
  // device set up
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  CHECK(cudaSetDevice(dev));

  // READ IN PARAMETERS
  double learn_rate;
  std::string cost_func;
  int epochs, size_batch;
  bool flag_digit, flag_host;
  std::vector <int> v_neurons;
  std::vector <std::string> v_layers;
  std::tie(flag_digit, flag_host, v_neurons, v_layers, cost_func, learn_rate, epochs, size_batch) = network_read_in();

  // ranges for the size batch
  std::pair<size_t, size_t> range_b = std::make_pair(10, 1000);
  size_t step_b = 10;

  // deciding the paths for the training data
  std::string path_train_data, path_train_label;
  if (flag_digit) {
    path_train_data  = "../data/training/train-images-idx3-ubyte";
    path_train_label = "../data/training/train-labels-idx1-ubyte";
  } else {
    path_train_data  = "../data/training/emnist-letters-train-images-idx3-ubyte";
    path_train_label = "../data/training/emnist-letters-train-labels-idx1-ubyte";
  }
  
  // read in the mnist data set
  unsigned char** data_set;
  unsigned char*  label_set;
  int n_images = 0, n_labels = 0, size_image = 0;
  std::tie(n_images, size_image, data_set) = read_mnist_data  (path_train_data);
  std::tie(n_labels, label_set)            = read_mnist_labels(path_train_label);

  // creating the neural network
  neural_network nn_name = create_neural_network(v_layers, v_neurons, cost_func, flag_host, learn_rate);
  
  // openning output file
  std::ofstream f_out;
  f_out.open(nn_name.produce_naming_short(v_neurons) + "_timing.dat");
  
  // loop over the batch size
  for (size_t size_b = range_b.first; size_b <= range_b.second; size_b += step_b) {

    // only print some interesting ones
    if (n_images % size_b != 0)
      continue;
    
    // create the network for the test -> different batch sizes
    // this resets the network -> not clean solution
    neural_network nn = create_neural_network(v_layers, v_neurons, cost_func, flag_host, learn_rate);
        
    // create matrices from input and output -> to work with normal network
    int n_batches = n_images / size_b;
    n_images = n_batches * size_b;
    std::vector <matrix> v_m_inp = create_matrices_input(data_set, size_image, n_images, size_b);
    std::vector <matrix> v_m_out = create_matrices_digits_target(label_set, n_labels, size_b, n_batches);
    // run and time the networks
    double mean_for, mean_back;
    std::tie(mean_for, mean_back) = time_network_test(nn, v_m_inp, v_m_out);
        
    // calculate flop for the networks
    double flop_for  = flop_neural_network_forward (v_neurons,  v_layers, size_b);
    double flop_back = flop_neural_network_backward(cost_func, v_neurons, v_layers, size_b);
        
    // print the header
    if (size_b == range_b.first) {
      f_out << std::setw(12) << "size_batch  "
	    << std::setw(12) << "for_time    "
	    << std::setw(12) << "back_time   "
	    << std::setw(12) << "forw_gflops "
	    << std::setw(12) << "back_gflops "
	    << std::endl;
    }
    
    // print data
    f_out << std::setw(12) << size_b
	  << std::setw(12) << mean_for
	  << std::setw(12) << mean_back
	  << std::setw(12) << flop_for  / mean_for  * 1e9
	  << std::setw(12) << flop_back / mean_back * 1e9
	  << std::endl;
  }

  // closing the data file, not really needed, gets closed anyway on destruction
  f_out.close();
}
