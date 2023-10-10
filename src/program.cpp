#include "program.h"
#include "evaluation.h"

program::program()
{
}

void program::append_expression(
    int expr_id,
    const char *op_name,
    const char *op_type,
    int inputs[],
    int num_inputs)
{
}

int program::add_op_param_double(
    const char *key,
    double value)
{
    return -1;
}

int program::add_op_param_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    return -1;
}

evaluation *program::build()
{
    return nullptr;
}
