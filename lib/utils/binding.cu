#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "matrix.h"
#include "neural_network.h"
#include "test_network.h"
#include "costs.h"
#include "relu.h"
#include "sigmoid.h"
#include "linear.h"
#include "softmax.h"
#include "mnist_reader.h"
#include "mse_cost.h"
#include "cce_cost.h"

// set up py namespace
namespace py = pybind11;


// module makro, file binding,module m
PYBIND11_MODULE(binding, m) {
  m.def("create_neural_network",&create_neural_network);
  // m.def("read_mnist_data",&read_mnist_data);
  // m.def("create_matrices_input",&create_matrices_input);

  //bind layer classes
  py::class_<layer>(m, "layer");


  py::class_<relu,layer>(m, "relu")
    .def(py::init<std::string>());
  
  py::class_<sigmoid,layer>(m, "sigmoid")
    .def(py::init<std::string>());
  
  py::class_<softmax,layer>(m, "softmax")
    .def(py::init<std::string>());
  
  py::class_<linear,layer>(m, "linear")
    .def(py::init<std::string, size_t, size_t>())
    .def("neurons_in",&linear::neurons_in)
    .def("neurons_out",&linear::neurons_out)
    .def("get_address",[](linear& l){ return reinterpret_cast<uint64_t>(&l);})
    .def("get_weights",&linear::get_weights_matrix)
    .def("get_biases",&linear::get_bias_matrix)
    .def("set_weights",&linear::set_weights_matrix)
    .def("set_biases",&linear::set_bias_matrix);

  // bid costs
  py::class_<costs>(m,"costs");
  
  py::class_<cce_cost,costs>(m, "cce_cost")
    .def(py::init<>())
    .def("cost",&cce_cost::cost)
    .def("dcost",&cce_cost::dcost);
  
  py::class_<mse_cost,costs>(m, "mse_cost")
    .def(py::init<>())
    .def("cost",&mse_cost::cost)
    .def("dcost",&mse_cost::dcost);
  
  py::class_<cce_soft_cost,costs>(m, "cce_soft_cost")
    .def(py::init<>())
    .def("cost",&cce_soft_cost::cost)
    .def("dcost",&cce_soft_cost::dcost);

  // bind network
  py::class_<neural_network>(m, "neural_network")
    .def(py::init<double,bool,bool>(),py::arg("learning_rate") = 0.01, py::arg("flag_host") = true, py::arg("flag_py") = true)
    .def("prop_forward",&neural_network::prop_forward)
    .def("prop_backward",&neural_network::prop_backward)
    .def("add_layer",&neural_network::add_layer,py::keep_alive<1, 2>())
    .def("set_cost",&neural_network::set_cost,py::keep_alive<1, 2>())
    .def("get_cost",&neural_network::get_cost)
    .def("get_layers",&neural_network::get_layers)
    .def("get_flag_host",&neural_network::get_flag_host);

  
    // bind matrix with buffer protocol
    py::class_<matrix>(m, "matrix", py::buffer_protocol())
       .def(py::init<size_t,size_t>())
       .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>>())
       .def_buffer([](matrix &m) -> py::buffer_info {
            return py::buffer_info(
                m.data_host.get(),                       /* Pointer to buffer */
                sizeof(double),                          /* Size of one scalar */
                py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
                2,                                       /* Number of dimensions */
                { m.rows(), m.cols() },                  /* Buffer dimensions */
                { sizeof(double) * m.cols(),             /* Strides (in bytes) for each index */
                  sizeof(double) }
            );
        })
        .def("rows", &matrix::rows)
        .def("cols", &matrix::cols)
        .def("alloc",&matrix::alloc)
        .def("copy_host_to_device",&matrix::copy_host_to_device)
        .def("copy_device_to_host",&matrix::copy_device_to_host);

}
