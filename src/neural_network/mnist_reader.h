/*
  MODULE TO READ IN THE MNIST DATA SET
  
  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 20.08.20
  TO-DO   :
  CAUTION : 
*/

#ifndef _MNIST_READER_H_
#define _MNIST_READER_H_

// c++ standard headers
#include <string>
#include <tuple>
#include <vector>

// own c++ headers
#include "matrix.h"

// helper functions
int reverse_int(int input);

// data and label read in functions
std::tuple<int, int, unsigned char**> read_mnist_data  (std::string f_name);
std::pair<int, unsigned char*>        read_mnist_labels(std::string f_name);

// transform raw data to matrix data
std::vector <matrix> create_matrices_input(unsigned char** data_set, int size_image, int n_images, int size_batch);
std::vector <matrix> create_matrices_digits_target(unsigned char* label_set, int n_labels, int size_batch, int n_batches);
std::vector <matrix> create_matrices_letters_target(unsigned char* label_set, int n_labels, int size_batch, int n_batches);

#endif // _MNIST_READER_H_
