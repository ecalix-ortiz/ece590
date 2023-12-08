#include "expression.h"
#include "tensor.h"

//Constructor
expression::expression(
    // arguments or parameters used for constructing an object
    int expr_id,
    const char *op_name,
    const char *op_type,
    int *inputs,
    int num_inputs)
    // returning values
    : expr_id_(expr_id),
    op_name_(op_name),
    op_type_(op_type),
    inputs_(inputs, inputs+num_inputs)
{
}

// result of evaluation stored here
void expression::add_op_param_double(
    const char *key,
    double value)
{
    op_param_[key] = tensor(value);
}

void expression::add_op_param_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    op_param_[key] = tensor(dim, shape, data);
}

// Getters
int expression::get_id() const{
    return expr_id_;
}

std::string expression::get_op_name() const{
    return op_name_;
}

std::string expression::get_op_type() const{
    return op_type_;
}

std::vector<int> expression::get_inputs() const{
    return inputs_;
}

tensor expression::get_op_param(std::string key) const{
    return op_param_.at(key);
}