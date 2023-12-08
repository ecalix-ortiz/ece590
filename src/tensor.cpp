#include "tensor.h"
#include <assert.h>

tensor::tensor()
    : data_(1,0)
{
}

tensor::tensor(
    double v)
    : data_(1,v)
{
}

tensor::tensor(
    int dim,
    size_t shape[],
    double data[])
    : shape_(shape, shape+dim)
{
    // calculate N as shape[0]*shape[1]*...*shape[dim-1]
    int N = 1;
    for (int i = 0; i < dim; i++) N = N*shape[i];
    data_.assign(data, data+N);
}

int tensor::get_dim() const
{
    return shape_.size();
}

double tensor::item() const 
{
    assert(shape_.empty());
    return data_[0];
}

double &tensor::item()
{
    assert(shape_.empty());
    return data_[0];
}

double tensor::at(size_t i) const
{
    assert(get_dim() == 1);
    assert(i < shape_[0]);
    return data_[i];
}

double tensor::at(
    size_t i,
    size_t j) const
{
    assert(get_dim() == 2);
    assert(i < shape_[0] && j < shape_[1]);
    return data_[(i*shape_[1])+j];
}

double tensor::at(
    size_t n,
    size_t c,
    size_t h,
    size_t w)
const {
    assert(get_dim() == 4);

    size_t N = shape_[0];
    size_t C = shape_[1];
    size_t H = shape_[2];
    size_t W = shape_[3];

    assert((n < N)
        && (c < C)
        && (h < H)
        && (w < W));
    
    return data_[n*C*H*W
        + c*H*W
        + h*W
        + w];
}

size_t *tensor::get_shape_array()
{
    return shape_.empty()? nullptr: &shape_[0];
}

double *tensor::get_data_array()
{
    return &data_[0];
}

long tensor::get_data_size() {
    return data_.size();
}

// Overloaded operators:
// tensor tensor::operator+(tensor &t) {
//     int dim = get_dim();
//     // if 1d, then add like normal
//     if (dim == 0) {
//         item() = item() + t.item();
//         return *this;
//     }

//     for (int i = 0; i < data_.size(); ++i)
//         data_[i] = data_[i] + t.data_[i];
    
//     return *this;
// }

// tensor tensor::operator*(tensor &t) {
//     int dim = get_dim();
//     // if 1d, then add like normal
//     if (dim == 0) {
//         item() = item() * t.item();
//         return *this;
//     }

//     for (int i = 0; i < data_.size(); ++i)
//         data_[i] = data_[i] * t.data_[i];
    
//     return *this;
// }

// // transposing
// tensor &tensor::transpose() {
//     assert(get_dim() == 2);

//     size_t row = shape_[0];
//     size_t col = shape_[1];
    
//     for (size_t r = 0; r < row; ++r)
//         for (size_t c = 0; c < col; ++c)
//             data_[c*row+r] = at(r, c);
    
//     shape_[0] = col;
//     shape_[1] = row;
//     return *this;
// }

// tensor tensor::transpose() const {
//     assert(get_dim() == 2);

//     size_t row = shape_[0];
//     size_t col = shape_[1];
//     size_t shape[] = {col, row};

//     double data[data_.size()];
//     for (size_t r = 0; r < row; ++r)
//         for (size_t c = 0; c < col; ++c)
//             data[c*row+r] = at(r, c);
    
//     return tensor(2, shape, data);
// }