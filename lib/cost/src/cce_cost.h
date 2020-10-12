#ifndef _CCE_COST_H_
#define _CCE_COST_H_

#include "matrix.h"
#include "costs.h"
#include "pw_comp.h"
#include "reduce.h"

//_______________________________________________________________________________________________
// categorical cross entropy cost
class cce_cost : public costs
{
public :
  // constructor / destructor
  cce_cost (void) {this->_type = CCE;};
  ~cce_cost(void) {};

  // functions to calculate
  double cost (matrix predict, matrix target, bool flag_host);
  matrix dcost(matrix predict, matrix target, matrix dy, bool flag_host);

  // operator overlaoding
  //friend std::ostream& operator <<(std::ostream& out, const cce_cost& cce);
  void print_out(std::ostream& out) const { out << "cce"; };
};


// ----------------------------------------------------------------------------------
// CCE and derivative on host and device
template<typename real_type>
struct d_cce_functor {
  __device__ __host__ real_type operator()(const real_type in,const real_type target) {
        
        if(!target) return 0;

        else return -target / in;    
    }
};

template<typename real_type>
struct summand_cce_functor {
  __device__ __host__ real_type operator()(const real_type in,const real_type target) {
        
        if(!target) return 0;

        else return -target * log(in);
    }
};

template<typename real_type>
real_type cce(const real_type *in,real_type *target,int size){
  return sum_func_array<real_type>(in,target,size,summand_cce_functor<real_type>())/size;
}

template<typename real_type>
void d_cce(const real_type *in, real_type *target, real_type *delta, int size){
  combine_pointwise<real_type>(in,target,delta,size,d_cce_functor<real_type>());
}

// // on device
template<typename real_type>
real_type cce_onDev(const real_type *dev_in, real_type *dev_target, int size){
  return sum_func_array_onDev<real_type>(dev_in,dev_target,size,summand_cce_functor<real_type>())/size;
}

template<typename real_type>
void d_cce_onDev(const real_type *dev_in, real_type *dev_target, real_type *dev_delta, int size){
  combine_pointwise_onDev<real_type>(dev_in,dev_target,dev_delta,size,d_cce_functor<real_type>());
}

#endif // _CCE_COST_H_