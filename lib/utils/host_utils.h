#ifndef _HOST_UTILS_H_
#define _HOST_UTILS_H_

#include "common_utils.h"

template <class real_type>
void apply_pointwise(const real_type *in, real_type *out, int size, pointwise_func<real_type> func)
{
    for (int i = 0; i < size; i++)
        out[i] = func(in[i]);
}

template <class real_type>
void hadamard_func_rhs(const real_type *lhs, real_type *rhs, real_type *res, int size, pointwise_func<real_type> func)
{
    for (int i = 0; i < size; i++)
        res[i] = lhs[i] * func(rhs[i]);
}

template <class real_type>
real_type get_max(const real_type *data, int length)
{
    real_type max = 0;
    for (int i = 0; i < length; i++)
        max = (data[i] > max ? data[i] : max);
    return max;
}

#endif // _HOST_UTILS_H_
