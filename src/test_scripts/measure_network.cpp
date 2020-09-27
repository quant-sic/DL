/*
  SCRIPT TO MEASURE COST AND ACCURACY DEPENDING ON EPOCHS

  AUTHOR  : FABIAN DECHENT / JANNIS SCHÜRMANN
  DATE    : 
  TO-DO   :
  CAUTION : 
*/
// c++ standard headers
#include <iostream>
#include <fstream>

// c++ own headers
#include "../timer.h"
#include "../run_scripts/run_network.h"
#include "../test_scripts/test_network.h"
#include "../neural_network/mnist_reader.h"
#include "../neural_network/neural_network.h"

//__________________________________________________________________________________________________
// MEASURE THE NETWORK COST AND ACCURACY DEPENDING ON THE EPOCH
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
  std::tie(flag_digit, flag_host, v_neurons, v_layers,
	   cost_func, learn_rate, epochs, size_batch) = network_read_in();
  
  // determintation of needed parameters
  std::string path_train_data, path_train_label, path_test_data, path_test_label;
  if (flag_digit) {
    path_train_data  = "../data/training/train-images-idx3-ubyte";
    path_train_label = "../data/training/train-labels-idx1-ubyte";
    path_test_data   = "../data/testing/t10k-images-idx3-ubyte";
    path_test_label  = "../data/testing/t10k-labels-idx1-ubyte";
  } else {
    path_train_data  = "../data/training/emnist-letters-train-images-idx3-ubyte";
    path_train_label = "../data/training/emnist-letters-train-labels-idx1-ubyte";
    path_test_data   = "../data/testing/emnist-letters-test-images-idx3-ubyte";
    path_test_label  = "../data/testing/emnist-letters-test-labels-idx1-ubyte";
  }
  
  // PROGRAM START
  // read in of training data
  int n_images_train, n_labels_train, size_image, n_batches_train;
  unsigned char** data_set; 
  unsigned char*  label_set;
  
  // reading in raw data training
  std::tie(n_images_train, size_image, data_set) = read_mnist_data  (path_train_data);
  std::tie(n_labels_train, label_set)            = read_mnist_labels(path_train_label);
  n_batches_train = n_images_train  / size_batch;
  n_images_train  = n_batches_train * size_batch;
  
  // create matrices from trainin set
  std::vector <matrix> v_m_out_train;
  std::vector <matrix> v_m_inp_train = create_matrices_input(data_set, size_image, n_images_train, size_batch);
  if (flag_digit)
    v_m_out_train = create_matrices_digits_target(label_set, n_labels_train, size_batch, n_batches_train);
  else
    v_m_out_train = create_matrices_letters_target(label_set, n_labels_train, size_batch, n_batches_train);
  
  // TESTING THE TRAINED NETWORK
  // read in raw data testing
  int n_images_test, n_labels_test, n_batches_test;
  std::tie(n_images_test, size_image, data_set) = read_mnist_data  (path_test_data);
  std::tie(n_labels_test, label_set)            = read_mnist_labels(path_test_label);
  n_batches_test = n_images_test  / size_batch;
  n_images_test  = n_batches_test * size_batch;
  
  // create test matrices from data set
  std::vector <matrix> v_m_out_test;
  std::vector <matrix> v_m_inp_test = create_matrices_input(data_set, size_image, n_images_test, size_batch);
  if (flag_digit)
    v_m_out_test = create_matrices_digits_target(label_set, n_labels_test, size_batch, n_batches_test);
  else
    v_m_out_test = create_matrices_letters_target(label_set, n_labels_test, size_batch, n_batches_test);
  
  
  // creating the network with layers
  neural_network nn = create_neural_network(v_layers, v_neurons, cost_func, flag_host, learn_rate);
  std::ofstream f_out;
  f_out.open(nn.produce_naming_short(v_neurons) + "_measure.dat");
  
  // TRAINING AND MEASURING ACCURACY AND COST
  matrix y;
  for (int epoch = 0; epoch < epochs; epoch++) {
    double cost = 0.;
    for (int batch = 0; batch < n_batches_train; batch++) {
      // propagate forward and backward
      y = nn.prop_forward(v_m_inp_train[batch]);
      nn.prop_backward(y, v_m_out_train[batch]);
      
      // calculate the cost
      cost += nn.get_cost()->cost(y, v_m_out_train[batch], flag_host);
    }

    // calculate predicts
    std::vector <matrix> v_predicts;
    for (int idx = 0; idx < v_m_inp_test.size(); idx++) {
      y = nn.prop_forward(v_m_inp_test[idx]);
      v_predicts.push_back(y);
    }

    // copy predicts to host
    if (!flag_host)
      for (int i = 0; i < v_predicts.size(); i++)
	v_predicts[i].copy_device_to_host();
    
    // print header
    if (epoch == 0)
      f_out << std::setw(12) << "Epoch"
	    << std::setw(12) << "Cost"
	    << std::setw(12) << "accuracy"
	    << std::endl;
    
    // write to file
    f_out << std::setw(12) << epoch << " "
	  << std::setw(12) << cost / n_batches_train
	  << std::setw(12) << compute_accuracy(v_predicts, v_m_out_test, v_neurons.back())
	  << std::endl;
  }
  
  // closing the output file
  f_out.close();
}
