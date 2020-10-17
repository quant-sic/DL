/*
  TEST THE NETWORK HEADER
  
  AUTHOR  : FABIAN DECHENT / JANNIS SCHÜRMANN
  DATE    :
  TO-DO   :
  CAUTION : 
*/

#include "timer.h"
#include "test_network.h"
#include "mse_cost.h"
#include "cce_cost.h"
#include "cce_soft_cost.h"

//_______________________________________________________________________________________________
// create a neural network from naming strings and sizes for the layers
neural_network create_neural_network(const std::vector <std::string>& v_layers,
				     const std::vector <int>&         v_neurons,
				     const std::string&               cost_name,
				     bool                             flag_host,
				     double                           learning_rate)
{
  // check for size of layers and neurons vector
  if ((v_layers.size() + 1) != v_neurons.size()) {
    std::cout << "(v_layers.size()+1) != v_neurons.size()" << std::endl;
    exit(-1);
  }
  
  // creating the basis instacne
  neural_network nn(learning_rate, flag_host);
  
  // deciding for the cost
  if (cost_name == "cce") {
    cce_cost* cce = new cce_cost();
    nn.set_cost(cce);
  } else if (cost_name == "mse") {
    mse_cost* mse = new mse_cost();
    nn.set_cost(mse);
  } else if (cost_name == "cce_soft") {
    cce_soft_cost* cce_soft = new cce_soft_cost();
    nn.set_cost(cce_soft);
  } else {
    std::cout << "The choosen cost option is not available.";
    exit(-1);
  }

  // creating the different layers
  for (int i = 0; i < v_layers.size(); i++) {
    // adding the linear layer to the network
    nn.add_layer(new linear("linear", v_neurons[i], v_neurons[i+1]));
    
    // adding activation
    if (v_layers[i] == "relu")
      nn.add_layer(new relu("relu"));
    else if (v_layers[i] == "sigmoid")
      nn.add_layer(new sigmoid("sigmoid"));
    else if (v_layers[i] == "softmax")
      nn.add_layer(new softmax("softmax"));
    else {
      std::cout << "The choosen activation layer is not available." << std::endl;
      exit(-1);
    }
  }

  return nn;
}

//_______________________________________________________________________________________________
// create a input sample to test the network -> which one is one hotencoded is choosen at random
std::vector <matrix> create_sample_one_hot(int  neurons,
					   int  n_batches,
					   int  size_batch)
{
  // decide for random position
  // srand(time(NULL));
  int position = rand() % neurons;
  
  // create single batch
  matrix mat_batch(size_batch, neurons);
  mat_batch.alloc();
  for (int row = 0; row < size_batch; row++) {
    for (int col = 0; col < neurons; col++) {
      if (col == 1)
	mat_batch[row*neurons+col] = 1.;
      else
	mat_batch[row*neurons+col] = 0;
    }
  }

  // create n batches
  std::vector <matrix> v_mat;
  for (int i = 0; i < n_batches; i++)
    v_mat.push_back(mat_batch);
  
  return v_mat;
}

//_______________________________________________________________________________________________
// run network with input and comput accuracy
// tests the one hot combination on the network
bool run_network_test(neural_network&       nn,
		      std::vector <matrix>& v_m_inp,
		      std::vector <matrix>& v_m_out,
		      int                   epochs)
{
  // check for vector sizes
  if (v_m_inp.size() != v_m_out.size()) {
    std::cout << "There should be the same nuber of batches for input and output.\n";
    exit(-1);
  }

  // moving data to device if needed
  if (!nn.get_flag_host())
    for (int i = 0; i < v_m_inp.size(); i++) {
      v_m_inp[i].copy_host_to_device();
      v_m_out[i].copy_host_to_device();
    }
  
  // running the neural network
  matrix y;
  double cost_val;
  std::ostringstream ret_out;
  ret_out << nn << "\n";
  for (int epoch = 0; epoch < epochs; epoch++) {
    cost_val = 0.;
    for (int batch = 0; batch < v_m_inp.size(); batch++) {
      // propagation
      y = nn.prop_forward(v_m_inp[batch]);
      nn.prop_backward(y, v_m_out[batch]);
      
      // cost calculation
      cost_val += nn.get_cost()->cost(y, v_m_out[batch], nn.get_flag_host());
    }
    if (epoch % 10 == 0)
      ret_out << "Epoch : " << epoch << " | cost : " << cost_val << "\n";
  }
  
  // calculation of predictions
  matrix prediction;
  std::vector <matrix> v_predicts;
  for (int batch = 0; batch < v_m_inp.size(); batch++) {
    prediction = nn.prop_forward(v_m_inp[batch]);
    v_predicts.push_back(prediction);
  }
    
  // moving predictions back to host if needed
  if (!nn.get_flag_host())
    for (int i = 0; i < v_m_inp.size(); i++)
      v_predicts[i].copy_device_to_host();

  // computing the accuracy
  double result =  compute_accuracy(v_predicts, v_m_out, 10);
  ret_out << "accuracy : " << result << "\n";
    
  // return values
  bool success = true;
  if (std::isnan(cost_val) or result != 1)
    success = false;
  if (!success)
    std::cout << ret_out.str() << std::endl;
  return success;
}

//________________________________________________________________________________________________
// time the neural network for each batch -> give back mean time for each batch in forward and
// bachward prop -> returns values in nanoseconds
// 1. value = mean_forward
// 2. value = mean_backward
std::pair<double, double> time_network_test(neural_network&       nn,
					    std::vector <matrix>& v_m_inp,
					    std::vector <matrix>& v_m_out)
{
  // check for vector sizes
  if (v_m_inp.size() != v_m_out.size()) {
    std::cout << "There should be the same nuber of batches for input and output.\n";
    exit(-1);
  }
  
  // moving data to device if needed
  if (!nn.get_flag_host())
    for (int i = 0; i < v_m_inp.size(); i++) {
      v_m_inp[i].copy_host_to_device();
      v_m_out[i].copy_host_to_device();
    }

  // running the neural network
  // timer for the complete calculation
  double mean_forward = 0., mean_backward = 0.;
  matrix y;
  for (int batch = 0; batch < v_m_inp.size(); batch++) {
    timer sw_for, sw_back;
    // propagation forward timing
    sw_for.start();
    y = nn.prop_forward(v_m_inp[batch]);
    sw_for.stop();
    mean_forward += sw_for.ns();

    // propagation backward timing
    sw_back.start();
    nn.prop_backward(y, v_m_out[batch]);
    sw_back.stop();
    mean_backward += sw_back.ns();
  }
  
  // return values
  mean_forward  /= v_m_inp.size();
  mean_backward /= v_m_inp.size();
  return std::make_pair(mean_forward, mean_backward);
}

//________________________________________________________________________________________________
// get the predicted number
// rewritten max function
int get_predicted_number(const matrix& mat,
			 const int&    idx_bat)
{
  int idx_max = 0;
  for (int i = 1; i < mat.cols(); i++)
    if (mat[idx_bat*mat.cols() + i] > mat[idx_bat*mat.cols() + idx_max])
      idx_max = i;
  
  return idx_max;
}

//________________________________________________________________________________________________
// compute the accuracy of the predicted things
double compute_accuracy(const std::vector <matrix>& v_predicts,
			const std::vector <matrix>& v_targets,
			int                         n_targets)
{
  // comapre sizes
  if (v_predicts.size() != v_targets.size()) {
    std::cout << "v_predicts.size() != v_targets.size()" << std::endl;
    std::cout << v_predicts.size() << " != " << v_targets.size() << std::endl;
    exit(-1);
  }
  
  // check the matrix sizes
  for (int idx = 0; idx < v_predicts.size(); idx++) {
    if (v_predicts[idx].rows() != v_targets[idx].rows()) {
      std::cout << __FUNCTION__ << std::endl;
      std::cout << "v_predicts[" << idx << "].rows() != v_targets[" << idx << "].rows()" << std::endl;
      std::cout << v_predicts[idx].rows() << " != " << v_targets[idx].rows() << std::endl;
      exit(-1);
    } else if (v_predicts[idx].cols() != v_targets[idx].cols()) {
      std::cout << __FUNCTION__ << std::endl;
      std::cout << "v_predicts[" << idx << "].cols() != v_targets[" << idx << "].cols()" << std::endl;
      std::cout << v_predicts[idx].cols() << " != " << v_targets[idx].cols() << std::endl;
      exit(-1);
    }
  }
  
  // compute accuracy
  int correct = 0;
  int digit_predict = -1;
  for (int idx_mat = 0; idx_mat < v_predicts.size(); idx_mat++) {
    
    // calculate correct ones for all predicts in one batch
    int n_batches = v_predicts[idx_mat].rows();
    for (int idx_bat = 0; idx_bat < n_batches; idx_bat++) {
      digit_predict = get_predicted_number(v_predicts[idx_mat], idx_bat);
      if (v_targets[idx_mat][idx_bat*n_targets + digit_predict] == 1)
	correct++;
    }
  }
  
  return (double) correct / (v_predicts.size() * v_predicts[0].rows()); 
}

//________________________________________________________________________________________________
// function to calc flops of matrix multiplication
size_t flop_mat_mul(size_t rows_lhs,
		    size_t cols_lhs,
		    size_t cols_rhs)
{
  return rows_lhs * cols_rhs * 2*cols_lhs; // not 2*cols_lhs - 1
}

//________________________________________________________________________________________________
// function to calc flops linear_forward
size_t flop_linear_forward(size_t neurons_in,
			   size_t neurons_out,
			   size_t size_batch)
{
  size_t flop = flop_mat_mul(size_batch, neurons_in, neurons_out);
  // adding the add along col flops
  return flop + size_batch * neurons_out;
}

//________________________________________________________________________________________________
// function to calc flops linear_forward
size_t flop_linear_backward(size_t neurons_in,
			    size_t neurons_out,
			    size_t size_batch)
{
  size_t flop = flop_mat_mul(size_batch, neurons_out, neurons_in);
  // update weights mat mul + mull add direct
  flop += flop_mat_mul(neurons_in, size_batch, neurons_out) + 2*neurons_in*neurons_out;

  // update add reduce dim + mull add direct
  flop += size_batch * neurons_out + 2*neurons_out;
  
  return flop;
}

//________________________________________________________________________________________________
// calc flop relu for / size comparisons
size_t flop_relu_forward(size_t size_batch,
			 size_t neurons_out) {
  return size_batch * neurons_out;
}

//________________________________________________________________________________________________
// calc flop relu for / size comparisons + size multiplications
size_t flop_relu_backward(size_t size_batch,
			  size_t neurons_out)
{
  return 2 * size_batch * neurons_out;
}

//________________________________________________________________________________________________
// size sigmoid forward
size_t flop_sigmoid_forward(size_t size_batch,
			    size_t neurons_out)
{
  return 3 * size_batch * neurons_out;
}

//________________________________________________________________________________________________
// size sigmoid backward
size_t flop_sigmoid_backward(size_t size_batch,
			     size_t neurons_out)
{
  return 4 * size_batch*neurons_out;
}

//________________________________________________________________________________________________
// flop softmax forward
size_t flop_softmax_forward(size_t size_batch,
			    size_t neurons_out)
{
  return 5 * size_batch * neurons_out; 
}

//________________________________________________________________________________________________
// flop softmax backward
size_t flop_softmax_backward(size_t size_batch,
			     size_t neurons_out)
{
  return size_batch * neurons_out * (neurons_out + 1) + flop_softmax_forward(size_batch, neurons_out); 
}

//________________________________________________________________________________________________
// d cost flop
size_t flop_dcce(size_t size_batch,
		 size_t neurons_out)
{
  return size_batch * neurons_out;
}

//________________________________________________________________________________________________
// d cost flop
size_t flop_dmse(size_t size_batch,
		 size_t neurons_out)
{
  return size_batch * neurons_out;
}

//________________________________________________________________________________________________
// d cost flop
size_t flop_dcce_soft(size_t size_batch,
		      size_t neurons_out)
{
  return flop_softmax_forward(size_batch, neurons_out) + size_batch * neurons_out;
}

//________________________________________________________________________________________________
// cal flop for neural network forwards
size_t flop_neural_network_forward(std::vector <int>         v_sizes,
				   std::vector <std::string> v_names,
				   size_t                    size_batch)
{
  // calculate the linear layer ppart
  size_t flop = 0;
  for (int idx = 0; idx < v_sizes.size() - 1; idx++)
    flop += flop_linear_forward(v_sizes[idx], v_sizes[idx+1], size_batch);
  
  // calculate activation part
  for (int idx = 1; idx < v_sizes.size(); idx++)
    if (v_names[idx-1] == "sigmoid")
      flop += flop_sigmoid_forward(size_batch, v_sizes[idx]);
    else if (v_names[idx-1] == "relu")
      flop += flop_relu_forward(size_batch, v_sizes[idx]);
    else if (v_names[idx-1] == "softmax")
      flop += flop_softmax_forward(size_batch, v_sizes[idx]);
  
  return flop;
}

//________________________________________________________________________________________________
// cal flop for neural network forwards
size_t flop_neural_network_backward(std::string               cost,
				    std::vector <int>         v_sizes,
				    std::vector <std::string> v_names,
				    size_t                    size_batch)
{
  // calculate the linear layer part
  size_t flop = 0;
  for (int idx = 0; idx < v_sizes.size() - 1; idx++)
    flop += flop_linear_backward(v_sizes[idx], v_sizes[idx+1], size_batch);

  // add the inital cost
  if (cost == "cce")
    flop += flop_dcce(size_batch, v_sizes.back());
  else if (cost == "mse")
    flop += flop_dmse(size_batch, v_sizes.back());
  else if (cost == "cce_soft")
    flop += flop_dcce_soft(size_batch, v_sizes.back());
  
  // calculate the activation part
  for (int idx = 1; idx < v_sizes.size(); idx++)
    if (v_names[idx-1] == "sigmoid")
      flop += flop_sigmoid_backward(size_batch, v_sizes[idx]);
    else if (v_names[idx-1] == "relu")
      flop += flop_relu_backward(size_batch, v_sizes[idx]);
    else if (v_names[idx-1] == "softmax")
      flop += flop_softmax_backward(size_batch, v_sizes[idx]);

  // return the number of flop
  return flop;
}
