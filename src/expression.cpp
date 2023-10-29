#include "expression.h"

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
    op_param_[key] = value;
}

void expression::add_op_param_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
}