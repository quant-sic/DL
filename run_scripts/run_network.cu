/*
  SIMPLE RUN FILE FOR THE NEURAL NETWORK
  
  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN 
  DATE    : 18.08.2020
  TO-DO   : 
  CAUTION : 
*/
// c++ standard headers
#include <iostream>

// c++ own headers
#include "timer.h"
#include "mnist_reader.h"
#include "neural_network.h"
#include "test_network.h"

//__________________________________________________________________________________________________
// TRAINING THE NEURAL NETWORK
int main(int argc, char** argv) {

  // device set up
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  CHECK(cudaSetDevice(dev));

  // READ IN PARAMETERS
  std::string flag_digits_letters;
  std::string path_train_data, path_train_label, path_test_data, path_test_label;
  std::cout << "____________________________________________________\n";
  std::cout << "NEURAL NETWORK\n";
  std::cout << "\n";
  while(true) {
    std::cout << "Should the Network be trained on letters or digits? (letters/digits)   : ";
    std::cin >> flag_digits_letters;
    if (flag_digits_letters == "letters") {
      path_train_data  = "data/training/emnist-letters-train-images-idx3-ubyte";
      path_train_label = "data/training/emnist-letters-train-labels-idx1-ubyte";
      path_test_data   = "data/testing/emnist-letters-test-images-idx3-ubyte";
      path_test_label  = "data/testing/emnist-letters-test-labels-idx1-ubyte";
      break;
    } else if (flag_digits_letters == "digits") {
      path_train_data  = "data/training/train-images-idx3-ubyte";
      path_train_label = "data/training/train-labels-idx1-ubyte";
      path_test_data   = "data/testing/t10k-images-idx3-ubyte";
      path_test_label  = "data/testing/t10k-labels-idx1-ubyte";
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
  
  // PROGRAM START
  // read in of training data
  int n_images, n_labels, size_image, n_batches;
  unsigned char** data_set; 
  unsigned char*  label_set;

  // starting the timing
  timer time_table;
  time_table.start();
  
  // reading in raw data
  std::tie(n_images, size_image, data_set) = read_mnist_data  (path_train_data);
  std::tie(n_labels, label_set)            = read_mnist_labels(path_train_label);
  n_batches = n_images  / size_batch;
  n_images  = n_batches * size_batch;
  
  // create matrices from input and output -> to work with normal network
  std::vector <matrix> v_m_out;
  std::vector <matrix> v_m_inp = create_matrices_input(data_set, size_image, n_images, size_batch);
  if (flag_digits_letters == "digits")
    v_m_out = create_matrices_digits_target(label_set, n_labels, size_batch, n_batches);
  else if (flag_digits_letters == "letters")
    v_m_out = create_matrices_letters_target(label_set, n_labels, size_batch, n_batches);
  
  // creating the network with layers
  neural_network nn = create_neural_network(v_layers, v_neurons, cost_function, flag_host, learning_rate);
  
  // move everything over to device if needed
  if (!flag_host)
    for (int i = 0; i < v_m_inp.size(); i++) {
      v_m_inp[i].copy_host_to_device();
      v_m_out[i].copy_host_to_device();
    }
  time_table.section("read and transform training data");
  
  // propagate network
  std::cout << "____________________________________________________\n";  
  std::cout << "TRAINING : " << std::endl;
  matrix y;
  for (int epoch = 0; epoch < epochs; epoch++) {
    double cost = 0.;
    for (int batch = 0; batch < n_batches; batch++) {
      // propagate forward and backward
      y = nn.prop_forward(v_m_inp[batch]);
      nn.prop_backward(y, v_m_out[batch]);
      
      // calculate the cost
      cost += nn.get_cost()->cost(y, v_m_out[batch], flag_host);
    }
    std::cout << "Epoch : " << std::setw(4) << epoch << " | cost : " << cost / n_batches << std::endl;
  }
  time_table.section("train neural network");

  // TESTING THE TRAINED NETWORK
  // read in the test data
  std::tie(n_images, size_image, data_set) = read_mnist_data  (path_test_data);
  std::tie(n_labels, label_set)            = read_mnist_labels(path_test_label);
  n_batches = n_images  / size_batch;
  n_images  = n_batches * size_batch;
  
  // create test matrices from data set
  v_m_inp = create_matrices_input(data_set, size_image, n_images, size_batch);
  if (flag_digits_letters == "letters")
    v_m_out = create_matrices_letters_target(label_set, n_labels, size_batch, n_batches);
  else if (flag_digits_letters == "digits")
    v_m_out = create_matrices_digits_target(label_set, n_labels, size_batch, n_batches);
  
  // compute the test m
  std::vector <matrix> v_predicts;
  for (int batch = 0; batch < v_m_inp.size(); batch++) {
    y = nn.prop_forward(v_m_inp[batch]);
    v_predicts.push_back(y);
  }
  
  // copy predicts to host
  if (!flag_host)
    for (int i = 0; i < v_predicts.size(); i++)
      v_predicts[i].copy_device_to_host();
  
  // compute the accuracy from the predictions of the training set
  std::cout << "accuracy : " << compute_accuracy(v_predicts, v_m_out, n_target_neurons) << std::endl;
  time_table.section("compare test set");
  
  // output time table
  std::cout << time_table << std::endl;
}
