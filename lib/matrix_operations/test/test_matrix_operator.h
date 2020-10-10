/*
  HEADER MODULE TEST MATRIX OPERATIONS

  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 11.08.2020
  TO-DO   :
  CAUTION : 1. when "hadamard" or "add" use : rows_lhs = cols_rhs
*/

#ifndef _TEST_MATRIX_OPERATOR_H_
#define _TEST_MATRIX_OPERATOR_H_

//___________________________________________________________________________________________________
// macro to print function name
#ifndef GET_FUNCTION_NAME
#define GET_FUNCTION_NAME(f) (#f)
#endif

#ifdef __cplusplus
extern "C" {
#endif

  // measure timings
  void time_matrix_operator_host  (const char* func_name,
				   void (f_matrix_operation) (const double* lhs, const double* rhs, int rows_lhs, int cols_rhs, int cols_lhs, double* res, int threads_block),
				   int rows_lhs,
				   int cols_rhs,
				   int cols_lhs,
				   int rep,
				   int max_threads_block);
  void time_matrix_operator_device(const char* func_name,
				   void (f_matrix_operation) (const double* lhs, const double* rhs, int rows_lhs, int cols_rhs, int cols_lhs, double* res, int threads_block),
				   int rows_lhs,
				   int cols_rhs,
				   int cols_lhs,
				   int rep,
				   int max_threads_block);

  // comparison of matrices
  void compare_host_device_operator(const char* func_name,
				    void (f_host)  (const double* lhs, const double* rhs, int rows_lhs, int cols_rhs, int cols_lhs, double* res, int threads_block),
				    void (f_device)(const double* lhs, const double* rhs, int rows_lhs, int cols_rhs, int cols_lhs, double* res, int threads_block),
				    int rows_lhs,
				    int cols_rhs,
				    int cols_lhs,
				    int threads_block);

  //__________________________________________________________________________________________________
  // smaller helper functions for this module
  double random_double         (double min, double max);
  void   create_random_matrix  (double* mat, int size, double min, double max);
  void   create_unit_matrix    (double* mat, int size, double value);
  void   print_matrix          (double* mat, int rows, int cols);
  void   compare_matrices_error_printless(double* res,double* lhs,double* rhs,int     size);
  void   compare_matrices_error(const char* comp_name, double* lhs, double* rhs, int rows, int cols);
  void   ONE_Matrix(double *mat,int dim,double value);

  //__________________________________________________________________________________________________
  // function wrappers for easy hadamard and add testing
  void matrix_hadamard_gpu_wrapper(const double* lhs, const double* rhs, int rows_lhs, int cols_rhs, int cols_lhs, double* res, int threads_block);
  void matrix_add_gpu_wrapper     (const double* lhs, const double* rhs, int rows_lhs, int cols_rhs, int cols_lhs, double* res, int threads_block);
  void matrix_hadamard_cpu_wrapper(const double* lhs, const double* rhs, int rows_lhs, int cols_rhs, int cols_lhs, double* res, int threads_block);
  void matrix_add_cpu_wrapper     (const double* lhs, const double* rhs, int rows_lhs, int cols_rhs, int cols_lhs, double* res, int threads_block);

#ifdef __cplusplus
}
#endif

#endif // _TEST_MATRIX_OPERATOR_
