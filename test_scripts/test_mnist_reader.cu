/*
  WRITE THE MNIST DATASET TO TXT FILE TO COMPARE WITH PYTHON
*/

#include <fstream>
#include <ostream>
#include "mnist_reader.h"

void write_mnist_to_file(std::string f_out, std::vector <matrix>& v_data, std::vector <matrix>& v_labels);

int main(int argc, char** argv) {

  // paths to files needed for this
  std::string path_data_set = "../data/training/train-images-idx3-ubyte";
  std::string path_label_set = "../data/training/train-labels-idx1-ubyte";
  std::string path_output = "mnist_reader_check.txt";
  
  // reading in the pure mnist data
  unsigned char** data_set;
  unsigned char* label_set;
  int size_image = 0;
  int n_images = 0;
  int n_labels = 0;
  std::tie(n_images, size_image, data_set) = read_mnist_data(path_data_set);
  std::tie(n_labels, label_set) = read_mnist_labels(path_label_set);

  std::vector <matrix> v_input = create_matrices_input(data_set, size_image, n_images, 1);
  std::vector <matrix> v_target = create_matrices_digits_target(label_set, n_labels, 1, 60000);
  
  write_mnist_to_file(path_output, v_input, v_target);
}

// write the labels to an txt file
void write_mnist_to_file(std::string f_name,
			 std::vector <matrix>& v_input,
			 std::vector <matrix>& v_target)
{
  std::ofstream f_out;
  f_out.open(f_name);

  for (int idx = 0; idx < v_input.size(); idx++) {
    // print the label number
    for (int l = 0; l < v_target[idx].cols(); l++)
      if (v_target[idx][l] == 1)
	f_out << l << " ";
    
    // print the label one hot
    for (int l = 0; l < v_target[idx].cols(); l++)
      f_out << v_target[idx][l] << " ";

    // print the mnist data as numbers
    for (int m = 0; m < v_input[idx].cols(); m++)
      f_out << v_input[idx][m] << " ";
    
    // next line
    f_out << "\n";
  }
}
