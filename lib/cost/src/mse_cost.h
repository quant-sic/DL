#ifndef _MSE_COST_H_
#define _MSE_COST_H_


#include "reduce.h"
#include "costs.h"
#include "pw_comp.h"
#include "matrix.h"


//_______________________________________________________________________________________________
// categorical cross entropy cost
class mse_cost : public costs
{
public :
  // constructor
  mse_cost (void) {this->_type = MSE;};
  ~mse_cost(void) {};

  // functions to calculate
  double cost (matrix predict, matrix target, bool flag_host);
  matrix dcost(matrix predict, matrix target, matrix dy, bool flag_host);

  // operator overloading
  //friend std::ostream& operator <<(std::ostream& out, const mse_cost& mse);
  void print_out(std::ostream& out) const { out << "mse"; };
};


// ----------------------------------------------------------------------------------
// MSE and derivative on host and device
template<typename real_type>
struct d_mse_functor {
  __device__ __host__ real_type operator()(const real_type in,const real_type target) { return  in - target; }
};

template<typename real_type>
struct summand_mse_functor {
  __device__ __host__ real_type operator()(const real_type in,const real_type target) { return  0.5*(in - target)*(in - target); }
};

template<typename real_type>
real_type mse(const real_type *in,real_type *target,int size){
  return sum_func_array<real_type>(in,target,size,summand_mse_functor<real_type>())/size;
}

template<typename real_type>
void d_mse(const real_type *in, real_type *target, real_type *delta, int size){
  combine_pointwise<real_type>(in,target,delta,size,d_mse_functor<real_type>());
}

// // on device
template<typename real_type>
real_type mse_onDev(const real_type *dev_in, real_type *dev_target, int size){
  return sum_func_array_onDev<real_type>(dev_in,dev_target,size,summand_mse_functor<real_type>())/size;
}

template<typename real_type>
void d_mse_onDev(const real_type *dev_in, real_type *dev_target, real_type *dev_delta, int size){
  combine_pointwise_onDev<real_type>(dev_in,dev_target,dev_delta,size,d_mse_functor<real_type>());
}



#endif // _MSE_COST_H_