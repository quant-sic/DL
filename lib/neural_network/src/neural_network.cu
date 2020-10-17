/*
  NEURAL NETWORK CLASS

  AUTHOR  : FABIAN DECHENT / JANNIS SCHÜRMANN
  DATE    : 18.08.2020
  TO-DO   :
  CAUTION :
*/

#include <sstream>

#include "costs.h"
#include "neural_network.h"
#include "layers.h"


//_______________________________________________________________________________________________
// constructor
neural_network::neural_network(double learning_rate,
			       bool   flag_host,
			       bool   flag_py) :
  learning_rate(learning_rate),
  flag_host(flag_host),
  flag_py(flag_py)
{
}

//_______________________________________________________________________________________________
// destructor
neural_network::~neural_network(void)
{
  if (!flag_py) {
    for (auto layer : v_layers)
      delete layer;
    delete _cost;
  }
}

//_______________________________________________________________________________________________
// forward propagation
// -> calculation of the forward propagation by each function for each layer
matrix neural_network::prop_forward(matrix x)
{
  matrix z = x;
  for (int idx_lay = 0; idx_lay < v_layers.size(); idx_lay++)
    z = v_layers[idx_lay]->prop_forward(z, flag_host);
  
  y = z;
  
  if (_cost->get_type() == CCE_SOFT)
    if (flag_host)
      softmax_activation_cpu(y.data_host.get(), y.data_host.get(), z.rows(), z.cols());
    else
      softmax_activation_onDev(y.data_device.get(), y.data_device.get(), z.rows(), z.cols());
  
  return y;
}

//_______________________________________________________________________________________________
// backward propagation
void neural_network::prop_backward(matrix predict,
				   matrix target)
{
  dy.alloc_if_not_allocated(predict.rows(), predict.cols());
  matrix error = _cost->dcost(predict, target, dy, flag_host);
  for (int idx_lay = v_layers.size()-1; idx_lay >= 0; idx_lay--)
    error = v_layers[idx_lay]->prop_backward(error, learning_rate, flag_host);
}

//_______________________________________________________________________________________________
// stream operator overloading
std::ostream& operator <<(std::ostream&         out,
			  const neural_network& nn)
{
  out << "____________________________________________________________________________________\n";
  out << "Neural Network : Layers      : ";
  std::vector <layer*> v_layers = nn.get_layers();
  for (auto layer : v_layers)
    out << *layer << " ";
  out << "\n";
  out << "               : Cost        : " << *nn.get_cost() << "\n";
  out << "               : host/device : ";
  if (nn.get_flag_host()) out << "host\n";
  else out << "device\n";

  return out;
}

//_______________________________________________________________________________________________
// creating a naming short for easy file name storage
// v_neurons is hardly accesable (cast into linear?, but information is already given
// in the functions that need this function)
std::string neural_network::produce_naming_short(std::vector <int>& v_neurons)
{
  std::string name = "nn_";

  // adding the layers to the short
  for (int idx = 0; idx < v_layers.size(); idx++)
    name += v_layers[idx]->get_name() + "_";

  // adding the cost to the short
  if (_cost->get_type() == MSE)
    name += "mse_";
  else if (_cost->get_type() == CCE)
    name += "cce_";
  else if (_cost->get_type() == CCE_SOFT)
    name += "cce_soft_";
  
  // adding the neuron numbers
  for (int idx = 0; idx < v_neurons.size(); idx++)
    name += std::to_string(v_neurons[idx]) + "_";

  // adding the flag_host
  if (flag_host)
    name += "host";
  else
    name += "device";
  
  // returning the name string
  return name;
}
