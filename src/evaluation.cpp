#include <assert.h>
#include "evaluation.h"

evaluation::evaluation(const std::vector<expression> &exprs)
    : result_(0)
{
}

void evaluation::add_kwargs_double(
    const char *key,
    double value)
{
}

void evaluation::add_kwargs_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
}

int evaluation::execute()
{
    return -1;
}

double &evaluation::get_result()
{
    return result_;
}
