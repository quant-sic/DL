/*
  MATRIX CLASS

  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 21.08.2020
  TO-DO   :
  CAUTION :
*/

#include "matrix.h"

//_________________________________________________________________________________________________
// empty constructor
matrix::matrix(void)
{
  _rows = 0;
  _cols = 0;
  _size = 0;
  data_device = nullptr;
  data_host   = nullptr;
  flag_host_alloc   = false;
  flag_device_alloc = false;
}
//_________________________________________________________________________________________________
// copy constructor
matrix::matrix(const matrix& m){
  // setting parameters
  _rows = m.rows();
  _cols = m.cols();
  _size = m.rows() * m.cols();

  // setting flags
  flag_device_alloc = m.get_flag_device_alloc();
  flag_host_alloc   = m.get_flag_host_alloc();

  // copy data
  // device
  if(flag_device_alloc){
    double* device_memory;
    CHECK(cudaMalloc((void**)&device_memory,_size*sizeof(double)));
    CHECK(cudaMemcpy(device_memory, m.data_device.get(), _size*sizeof(double), cudaMemcpyDeviceToDevice));
    data_device = std::shared_ptr<double>(device_memory, [&](double* ptr) { CHECK(cudaFree(ptr)) ;});
  }else{
    data_device = nullptr;
  }

  // host
  if(flag_host_alloc){
    data_host = std::shared_ptr<double>(new double[_size], [&](double* ptr) { delete [] ptr; });
    memcpy(data_host.get(),m.data_host.get(),_size*sizeof(double));
  }else{
    data_host   = nullptr;
  }
}




//_________________________________________________________________________________________________
// constructor / standard constructor
matrix::matrix(size_t rows,
	       size_t cols)
{
  // setting parameters
  _rows = rows;
  _cols = cols;
  _size = _rows * _cols;

  // setting flags
  flag_device_alloc = false;
  flag_host_alloc   = false;

  // data pointers
  data_device = nullptr;
  data_host   = nullptr;
}

//_________________________________________________________________________________________________
// python constructor
matrix::matrix(py::array_t<double, py::array::c_style | py::array::forcecast> array)
{
  py::buffer_info buf = array.request();

  // setting parameters
  _rows = buf.shape[0];
  _cols = buf.shape[1];
  _size = _rows * _cols;

  // setting flags
  flag_device_alloc = false;
  flag_host_alloc   = true;

  // data pointers
  data_device = nullptr;
  data_host = std::shared_ptr<double>((double*)buf.ptr,[&](double* ptr) {});
}

//_________________________________________________________________________________________________
// allocate matrix memory on the host
void matrix::alloc_host(void)
{
  if (!flag_host_alloc) {
    data_host = std::shared_ptr<double>(new double[_size], [&](double* ptr) { delete [] ptr; });
    flag_host_alloc = true;
  }
}

//_________________________________________________________________________________________________
// allocate matrix memory on device
void matrix::alloc_device(void)
{
  if (!flag_device_alloc) {
    double* device_memory = nullptr;
    CHECK(cudaMalloc((void**)&device_memory, _size*sizeof(double)));
    data_device = std::shared_ptr<double>(device_memory, [&](double* ptr) { CHECK(cudaFree(ptr)) ;});
    flag_device_alloc = true;
  }
}

//_________________________________________________________________________________________________
// copy matrix memory from device to host
void matrix::copy_device_to_host(void)
{
  // allocate host memory if needed
  if (!flag_host_alloc)
    alloc_host();

  // check if the matrix is allocated on device -> useless otherwise to copy
  if (!flag_device_alloc) {
    std::cout << __FUNCTION__ << "\n";
    std::cout << "Memory not allocated on device. Cannot move from device to host.\n";
  }
  
  // copy the actual memory around
  CHECK(cudaMemcpy(data_host.get(), data_device.get(), _size*sizeof(double), cudaMemcpyDeviceToHost));
}

//_________________________________________________________________________________________________
// copy matrix memory from host to device
void matrix::copy_host_to_device(void)
{
  // allocate host memory if needed
  if (!flag_device_alloc)
    alloc_device();

  // check if the matrix is allocated on host -> useless otherwise to copy
  if (!flag_host_alloc) {
    std::cout << __FUNCTION__ << "\n";
    std::cout << "Memory not allocated on host. Cannot move from host to device.\n";
  }

  // copy the actual memory around
  CHECK(cudaMemcpy(data_device.get(), data_host.get(), _size*sizeof(double), cudaMemcpyHostToDevice));
}

//_________________________________________________________________________________________________
// matrix fucntion to allocate memory if not allocated
void matrix::alloc_if_not_allocated(size_t rows,
				    size_t cols)
{
  if (!flag_device_alloc and !flag_host_alloc) {
    _rows = rows;
    _cols = cols;
    _size = _rows * _cols;
    alloc();
  }
}

//_________________________________________________________________________________________________
// allocate storage
void matrix::alloc(void)
{
  alloc_host();
  alloc_device();
}

//_________________________________________________________________________________________________
// stream operator overloading -> rather print matrix?
std::ostream& operator <<(std::ostream& out,
			  const matrix& mat)
{
  // output of defining parameeters
  out << "rows         : " << mat._rows << std::endl;
  out << "cols         : " << mat._cols << std::endl;
  out << "size         : " << mat._size << std::endl;
  out << "host_alloc   : ";
  if (mat.flag_host_alloc) std::cout << "true" << std::endl;
  else std::cout << "false" << std::endl;
  out << "device_alloc : ";
  if (mat.flag_device_alloc) std::cout << "true" << std::endl;
  else std::cout << "false" << std::endl;

  return out;
}

//_________________________________________________________________________________________________
// acces operator host data
double& matrix::operator[](const int idx)
{
  return data_host.get()[idx];
}

//_________________________________________________________________________________________________
// const acces operator host data
const double& matrix::operator[](const int idx) const
{
  return data_host.get()[idx];
}

//_________________________________________________________________________________________________
// print out device data
void output_device_matrix(matrix& mat)
{
  mat.copy_device_to_host();
  double* data = mat.data_host.get();
  for (int row = 0; row < mat.rows(); row++) {
    for (int col = 0; col < mat.cols(); col++)
      std::cout << data[row*mat.cols() + col] << " ";
    std::cout << "\n";
  }
}

//_________________________________________________________________________________________________
// print out host data
void output_host_matrix(matrix& mat)
{
  double* data = mat.data_host.get();
  for (int row = 0; row < mat.rows(); row++) {
    for (int col = 0; col < mat.cols(); col++)
      std::cout << data[row*mat.cols() + col] << " ";
    std::cout << "\n";
  }
}

//_________________________________________________________________________________________________
// print out matrix depeneidng on flag
void output_matrix(matrix& mat,
		   bool    flag_host)
{
  if (flag_host)
    output_host_matrix(mat);
  else
    output_device_matrix(mat);
}
