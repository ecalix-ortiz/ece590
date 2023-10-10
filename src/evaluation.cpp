#include <assert.h>
#include "evaluation.h"

evaluation::evaluation(const std::vector<expression> &exprs)
    : result_(0)
{
    for (auto &expr: exprs)
        if (expr.expr_id_>result_)
            result_=expr.expr_id_;
}

void evaluation::add_kwargs_double(
    const char *key,
    double value)
{
    // access keyword arguments using key
    kwargs_[key].push_back(value);
    // printf("value of %s is : %f\n", key, kwargs_[key]);
}

void evaluation::add_kwargs_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    size_t n=1;
    for(int i=0; i<dim; ++i) n*=shape[i];
    kwargs_[key].assign(data,data+n);
    // printf("value of %s is : {", key);
    // for (auto iter:kwargs_[key]) printf(iter);
}

int evaluation::execute()
{
    // printf("value of %s is : %f\n", key, kwargs_["a"]);
    
    return 0;
}

double &evaluation::get_result()
{
    return result_;
}
