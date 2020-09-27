/*
  MODULE TO READ IN THE MNIST DATA SET
  
  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 20.08.20
  TO-DO   :
  CAUTION : 
*/

// c++ standard headers
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>

// own headers
#include "mnist_reader.h"

//________________________________________________________________________________________________
// reverse a int into the real integer
int reverse_int(int input)
{
  unsigned char c1, c2, c3, c4;
  c1 = input         & 255;
  c2 = (input >> 8)  & 255;
  c3 = (input >> 16) & 255;
  c4 = (input >> 24) & 255;

  return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
};

//________________________________________________________________________________________________
// data read in
// returns the tuple 1. number of images
//                   2. image size
//                   3. data_set pointer
std::tuple<int, int, unsigned char**> read_mnist_data(std::string f_name)
{
  // opening input file
  std::ifstream f_inp(f_name, std::ios::binary);
  if (!f_inp) {
    std::cout << __FUNCTION__ << "\n";
    std::cout << "File " << f_name << "could not be openend.\n";
    exit(-1);
  }

  // read in magic number and check it
  int magic_number = 0;
  f_inp.read((char *)&magic_number, sizeof(magic_number));
  magic_number = reverse_int(magic_number);
  if (magic_number != 2051) {
    std::cout << __FUNCTION__ << "\n";
    std::cout << "Invalid magic number.";
    exit(-1);
  }
  
  // read in image parameters
  int n_rows = 0, n_cols = 0, n_images = 0, size_images = 0;
  f_inp.read((char *)&n_images, sizeof(n_images));
  n_images = reverse_int(n_images);
  f_inp.read((char *)&n_rows, sizeof(n_rows));
  n_rows = reverse_int(n_rows);
  f_inp.read((char *)&n_cols, sizeof(n_cols));
  n_cols = reverse_int(n_cols);
  size_images = n_rows * n_cols;
  
  // read in image data
  unsigned char** data_set = new unsigned char*[n_images]; 
  for (int i = 0; i < n_images; i++) {
    data_set[i] = new unsigned char[size_images];
    f_inp.read((char *)data_set[i], size_images);
  }

  // return the dataset and parameters
  return std::make_tuple(n_images, size_images, data_set);
}

//________________________________________________________________________________________________
// label read in
// returns the tuple 1. number of labels
//                   2. label set pointer
std::pair<int, unsigned char*> read_mnist_labels(std::string f_name)
{
  // openening input file
  std::ifstream f_inp(f_name, std::ios::binary);
  if (!f_inp) {
    std::cout << __FUNCTION__ << "\n";
    std::cout << "File " << f_name << "could not be openend.\n";
    exit(-1);
  }

  // read in magic number and check it
  int magic_number = 0;
  f_inp.read((char *)&magic_number, sizeof(magic_number));
  magic_number = reverse_int(magic_number);
  if (magic_number != 2049) {
    std::cout << __FUNCTION__ << "\n";
    std::cout << "Invalid magic number.";
    exit(-1);
  }
  
  // read number of labels
  int n_labels = 0;
  f_inp.read((char *)&n_labels, sizeof(n_labels));
  n_labels = reverse_int(n_labels);
  
  // read in label data
  unsigned char* label_set = new unsigned char[n_labels]; 
  for (int i = 0; i < n_labels; i++) {
    f_inp.read((char *)&label_set[i], 1);
  }
  
  // return the labels and number of labels
  return std::make_pair(n_labels, label_set);
}

//________________________________________________________________________________________________
// create the input matrices from the input data
// slow !!!
std::vector <matrix> create_matrices_input(unsigned char** data_set,
					   int             size_image,
					   int             n_images,
					   int             size_batch)
{
  // flatten and normalize the data set
  std::vector <matrix> v_matrices;
  for (int image = 0; image < n_images; image += size_batch) {
    // creating matrix for a batch of images
    matrix mat_batch(size_batch, size_image);
    mat_batch.alloc();
    for (int batch = 0; batch < size_batch; batch++) {
      unsigned char* d_image = data_set[image+batch];
      for (int ele = 0; ele < size_image; ele++)
	mat_batch[batch*size_image + ele] = (double) 1.0/255 * d_image[ele];
    }
    v_matrices.push_back(mat_batch);
  }

  // copy the data to the device directly
  for (int idx = 0; idx < v_matrices.size(); idx++)
    v_matrices[idx].copy_host_to_device();
  
  //free(data_out);
  return v_matrices;
}

//________________________________________________________________________________________________
// create the input matrices from the input data
std::vector <matrix> create_matrices_digits_target(unsigned char* label_set,
						   int            n_labels,
						   int            size_batch,
						   int            n_batches)
{
  // creating the matrix vector
  std::vector <matrix> v_labels_matrix;
  int n_digits = 10;
  for (int idx = 0; idx < n_labels; idx += size_batch) {
    matrix mat_batch(size_batch, n_digits);
    mat_batch.alloc();
    for (int batch = 0; batch < size_batch; batch++) {
      for (int digit = 0; digit < n_digits; digit++)
	mat_batch[batch*n_digits + digit] = 0.;
      mat_batch[batch*n_digits + (int)label_set[idx+batch]] = 1.;
    }
    v_labels_matrix.push_back(mat_batch);
  }

  // remove the last element if needed
  while(v_labels_matrix.size() > n_batches)
    v_labels_matrix.pop_back();
  
  // copy the data to the device directly
  for (int idx = 0; idx < v_labels_matrix.size(); idx++)
    v_labels_matrix[idx].copy_host_to_device();
  
  return v_labels_matrix;
}

//________________________________________________________________________________________________
// create the input matrices from the input data
std::vector <matrix> create_matrices_letters_target(unsigned char* label_set,
						    int            n_labels,
						    int            size_batch,
						    int            n_batches)
{
  // creating the matrix vectorb
  std::vector <matrix> v_labels_matrix;
  int n_letters = 26;
  for (int idx = 0; idx < n_labels; idx += size_batch) {
    matrix mat_batch(size_batch, n_letters);
    mat_batch.alloc();
    for (int batch = 0; batch < size_batch; batch++) {
      for (int digit = 0; digit < n_letters; digit++)
	mat_batch[batch*n_letters + digit] = 0.;
      if (label_set[idx+batch] == 1 or label_set[idx+batch] == 65 or label_set[idx+batch] == 97) {
	mat_batch[batch*n_letters + 0] = 1.;
      } else if (label_set[idx+batch] == 2  or label_set[idx+batch] == 66 or label_set[idx+batch] == 98) {
	mat_batch[batch*n_letters + 1] = 1.;
      } else if (label_set[idx+batch] == 3  or label_set[idx+batch] == 67 or label_set[idx+batch] == 99) {
	mat_batch[batch*n_letters + 2] = 1.;
      } else if (label_set[idx+batch] == 4  or label_set[idx+batch] == 68 or label_set[idx+batch] == 100) {
	mat_batch[batch*n_letters + 3] = 1.;
      } else if (label_set[idx+batch] == 5  or label_set[idx+batch] == 69 or label_set[idx+batch] == 101) {
	mat_batch[batch*n_letters + 4] = 1.;
      } else if (label_set[idx+batch] == 6  or label_set[idx+batch] == 70 or label_set[idx+batch] == 102) {
	mat_batch[batch*n_letters + 5] = 1.;
      } else if (label_set[idx+batch] == 7  or label_set[idx+batch] == 71 or label_set[idx+batch] == 103) {
	mat_batch[batch*n_letters + 6] = 1.;
      } else if (label_set[idx+batch] == 8  or label_set[idx+batch] == 72 or label_set[idx+batch] == 104) {
	mat_batch[batch*n_letters + 7] = 1.;
      } else if (label_set[idx+batch] == 9  or label_set[idx+batch] == 73 or label_set[idx+batch] == 105) {
	mat_batch[batch*n_letters + 8] = 1.;
      } else if (label_set[idx+batch] == 10 or label_set[idx+batch] == 74 or label_set[idx+batch] == 106) {
	mat_batch[batch*n_letters + 9] = 1.;
      } else if (label_set[idx+batch] == 11 or label_set[idx+batch] == 75 or label_set[idx+batch] == 107) {
	mat_batch[batch*n_letters +10] = 1.;
      } else if (label_set[idx+batch] == 12 or label_set[idx+batch] == 76 or label_set[idx+batch] == 108) {
	mat_batch[batch*n_letters +11] = 1.;
      } else if (label_set[idx+batch] == 13 or label_set[idx+batch] == 77 or label_set[idx+batch] == 109) {
	mat_batch[batch*n_letters +12] = 1.;
      } else if (label_set[idx+batch] == 14 or label_set[idx+batch] == 78 or label_set[idx+batch] == 110) {
	mat_batch[batch*n_letters +13] = 1.;
      } else if (label_set[idx+batch] == 15 or label_set[idx+batch] == 79 or label_set[idx+batch] == 111) {
	mat_batch[batch*n_letters +14] = 1.;
      } else if (label_set[idx+batch] == 16 or label_set[idx+batch] == 80 or label_set[idx+batch] == 112) {
	mat_batch[batch*n_letters +15] = 1.;
      } else if (label_set[idx+batch] == 17 or label_set[idx+batch] == 81 or label_set[idx+batch] == 113) {
	mat_batch[batch*n_letters +16] = 1.;
      } else if (label_set[idx+batch] == 18 or label_set[idx+batch] == 82 or label_set[idx+batch] == 114) {
	mat_batch[batch*n_letters +17] = 1.;
      } else if (label_set[idx+batch] == 19 or label_set[idx+batch] == 83 or label_set[idx+batch] == 115) {
	mat_batch[batch*n_letters +18] = 1.;
      } else if (label_set[idx+batch] == 20 or label_set[idx+batch] == 84 or label_set[idx+batch] == 116) {
	mat_batch[batch*n_letters +19] = 1.;
      } else if (label_set[idx+batch] == 21 or label_set[idx+batch] == 85 or label_set[idx+batch] == 117) {
	mat_batch[batch*n_letters +20] = 1.;
      } else if (label_set[idx+batch] == 22 or label_set[idx+batch] == 86 or label_set[idx+batch] == 118) {
	mat_batch[batch*n_letters +21] = 1.;
      } else if (label_set[idx+batch] == 23 or label_set[idx+batch] == 87 or label_set[idx+batch] == 119) {
	mat_batch[batch*n_letters +22] = 1.;
      } else if (label_set[idx+batch] == 24 or label_set[idx+batch] == 88 or label_set[idx+batch] == 120) {
	mat_batch[batch*n_letters +23] = 1.;
      } else if (label_set[idx+batch] == 25 or label_set[idx+batch] == 89 or label_set[idx+batch] == 121) {
	mat_batch[batch*n_letters +24] = 1.;
      } else if (label_set[idx+batch] == 26 or label_set[idx+batch] == 90 or label_set[idx+batch] == 122) {
	mat_batch[batch*n_letters +25] = 1.;
      }
    }
    v_labels_matrix.push_back(mat_batch);
  }

  // remove the last element if needed
  while(v_labels_matrix.size() > n_batches)
    v_labels_matrix.pop_back();
  
  // copy the data to the device directly
  for (int idx = 0; idx < v_labels_matrix.size(); idx++)
    v_labels_matrix[idx].copy_host_to_device();
  
  return v_labels_matrix;
}
