#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>

class tensor {
public:
    tensor(); // scalar 0
    explicit tensor(double v); // scalar v
    tensor(int dim, size_t shape[], double data[]); // from C

    // Accessors
    int get_dim() const;
    // scalar only
    double item() const;
    double &item();
    // for tensors
    double at(size_t i) const;
    double at(size_t i, size_t j) const;
    double at(size_t n, size_t c, size_t h, size_t w) const;
    
    // Passing tensor Back to C Code
    size_t *get_shape_array();
    double *get_data_array();
    long get_data_size();

    // Overloaded operators
    // tensor operator+(tensor &t); // +
    // tensor operator*(tensor &t); // *

    // // Transposing
    // tensor transpose() const;
    // tensor &transpose();
private:
    std::vector<size_t> shape_;
    std::vector<double> data_;
}; // class tensor

#endif // TENSOR_H