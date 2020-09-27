/*
  LINEAR LAYER CLASS

  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 19.08.2020
  TO-DO   : 1. change random number generation on device? for weights?
  CAUTION :
*/

#include "linear.h"
#include "../common.h"

//_________________________________________________________________________________________________
// constructor
linear::linear(std::string name,
	       size_t      neurons_in,
	       size_t      neurons_out) : w(neurons_in, neurons_out), b(neurons_out, 1), helper_storage(neurons_in, neurons_out)
{
  this->_type = LINEAR;
  this->_name = name;
  this->_neurons_in = neurons_in;
  this->_neurons_out = neurons_out;
  w.alloc();
  b.alloc();
  init_bias_zeros();
  init_weights_random();
  helper_storage.alloc();

  // initalising the number generator
  gen.seed(time(NULL));
}

//_________________________________________________________________________________________________
// destructor
linear::~linear(void)
{
}

// ________________________________________________________________________________________________
// set weights and biases
void linear::set_weights_matrix(matrix _w,
				bool   flag_host)
{
  w = _w;
  if (!flag_host)
    w.copy_host_to_device();
}

void linear::set_bias_matrix(matrix _b,
			     bool   flag_host)
{
  b = _b;
  if (!flag_host)
    b.copy_host_to_device();
}

//_________________________________________________________________________________________________
// iniatlize the weights randomly
void linear::init_weights_random(void)
{
  // init host data
  // std::default_random_engine gen;
  double limit = std::sqrt(6./(double)(_neurons_in +_neurons_out));
  std::uniform_real_distribution<double> n_dist(-limit, limit);
  for (size_t row = 0; row < w.rows(); row++)
    for (size_t col = 0; col < w.cols(); col++)
      w[row*w.cols() + col] = n_dist(gen);
  
  // copy data to device
  w.copy_host_to_device();
}

//_________________________________________________________________________________________________
// init the bias with zeros
void linear::init_bias_zeros(void)
{
  // init host data
  for (size_t idx = 0; idx < b.size(); idx++)
    b.data_host.get()[idx] = 0.;

  // copy data to device
  b.copy_host_to_device();
}

//_________________________________________________________________________________________________
// propagation forward
matrix& linear::prop_forward(matrix& a,
			     bool    flag_host)
{
  // check for size
  if (w.rows() != a.cols()) {
    std::cout << __FUNCTION__ << "\n";
    std::cout << "w.rows() != a.cols() (" << w.rows() << "!=" << a.cols() << ")" << std::endl;
    exit(-1);
  }

  this->a = a;
  z.alloc_if_not_allocated(a.rows(), w.cols());
  calc_store_layer_output(a, flag_host);
  return z;
}

//_________________________________________________________________________________________________
// calculate and store the ölayer output // w(neurons_in x neurons_out); a(batchsize x neurons_in)
void linear::calc_store_layer_output(matrix& a,
				     bool    flag_host)
{
  if (flag_host)
    linear_forward_cpu(w.data_host.get(), a.data_host.get(),
		       z.data_host.get(), b.data_host.get(),
		       w.rows(), w.cols(), a.rows(), a.cols()); // w(neurons_in x neurons_out); a(batchsize x neurons_in)
  else
    linear_forward_gpu(w.data_device.get(), a.data_device.get(),
		       z.data_device.get(), b.data_device.get(),
		       w.rows(), w.cols(), a.rows(), a.cols());
}

//_________________________________________________________________________________________________
// back propagation
matrix& linear::prop_backward(matrix& dz,
			      double  learning_rate,
			      bool    flag_host)
{
  da.alloc_if_not_allocated(a.rows(), a.cols());
  calc_store_backprop_error(dz, flag_host);
  update_bias   (dz, learning_rate, flag_host);
  update_weights(dz, learning_rate, flag_host);
  return da;
}

//_________________________________________________________________________________________________
// compute and stpore backprop error // w(neurons_in x neurons_out); dz(batchsize x neurons_out)
void linear::calc_store_backprop_error(matrix& dz,
				       bool    flag_host)
{
  if (flag_host)
    linear_backprop_cpu(w.data_host.get(), dz.data_host.get(), da.data_host.get(),
			w.rows(), w.cols(), dz.rows(), dz.cols());
  else
    linear_backprop_gpu(w.data_device.get(), dz.data_device.get(), da.data_device.get(),
			w.rows(), w.cols(), dz.rows(), dz.cols());
}

//_________________________________________________________________________________________________
// update the weights
void linear::update_weights(matrix& dz,
			    double  learning_rate,
			    bool    flag_host)
{
  if (flag_host)
    linear_update_weights_cpu(dz.data_host.get(), a.data_host.get(), w.data_host.get(),
			      dz.rows(), dz.cols(), a.rows(), a.cols(), learning_rate);
  else
    linear_update_weights_gpu(dz.data_device.get(), a.data_device.get(), w.data_device.get(),
			      dz.rows(), dz.cols(), a.cols(), learning_rate, helper_storage.data_device.get());
}

//_________________________________________________________________________________________________
// update the bias
void linear::update_bias(matrix& dz,
			 double  learning_rate,
			 bool    flag_host)
{
  if (flag_host)
    linear_update_bias_cpu(dz.data_host.get(), b.data_host.get(),
			   dz.rows(), dz.cols(), learning_rate);
  else
    linear_update_bias_gpu(dz.data_device.get(), b.data_device.get(),
			   dz.rows(), dz.cols(), learning_rate, helper_storage.data_device.get());
}
