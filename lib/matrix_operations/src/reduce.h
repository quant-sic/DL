#ifndef _REDUCE_H_
#define _REDUCE_H_




template<typename real_type>
extern void add_reduce_dim_cpu(const real_type* mat_in,real_type *vec_out, int rows,int cols, int dim_red,int dim_vec);

template<typename real_type>
extern __global__ void add_reduce_rows_kernel(const real_type * __restrict__  dev_mat_in, real_type * __restrict__  dev_vec_out, const int rows, const int cols);

template<typename real_type>
extern __global__ void add_reduce_cols_kernel(const real_type* dev_mat_in,real_type *dev_vec_out, int rows,int cols);

template<typename real_type>
extern void add_reduce_dim_onDev(const real_type* dev_mat_in,real_type *dev_vec_out, int rows,int cols, int dim_red,int size_vec);

template <class real_type>
extern real_type get_max(const real_type *data, int length);

template<typename real_type>
extern real_type get_max_onDev(const real_type *dev_in,int size);

#endif // _REDUCE_H