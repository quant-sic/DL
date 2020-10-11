#ifndef _COMMON_UTILS_H_
#define _COMMON_UTILS_H_

template<class real_type>
struct exp_sub_val_functor {
  exp_sub_val_functor(real_type val) : _val(val) {}  // Constructor

  __device__ __host__ real_type operator()(const real_type x) { return  exp( x-_val); }

  private:
    real_type _val;
};

template<class real_type>
struct scale_functor {
  scale_functor(real_type val) : _val(val) {}  // Constructor
  __device__ __host__ real_type operator()(const real_type x) { return  _val*x; }

  private:
    real_type _val;
};

template<class real_type>
struct add_functor {
  __device__ __host__ real_type operator()(const real_type x,const real_type y) { return  x+y; }
};

template<class real_type>
struct sub_functor {
  __device__ __host__ real_type operator()(const real_type x,const real_type y) { return  x-y; }
};

template<class real_type>
struct mul_functor {
  __device__ __host__ real_type operator()(const real_type x,const real_type y) { return  x*y; }
};

template<class real_type>
struct div_functor {
  __device__ __host__ real_type operator()(const real_type x,const real_type y) { return  x/y; }
};

#endif // _COMMON_UTILS_H_