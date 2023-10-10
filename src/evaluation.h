#ifndef EVALUATION_H
#define EVALUATION_H

#include "expression.h"

class evaluation
{
public:
    evaluation(const std::vector<expression> &exprs);

    void add_kwargs_double(
        const char *key,
        double value);

    void add_kwargs_ndarray(
        const char *key,
        int dim,
        size_t shape[],
        double data[]);

    // return 0 for success
    int execute();

    // return the variable computed by the last expression
    double &get_result();

private:
    double result_;

    // ----My Code---
    // std::map<std::string, double> kwargs_;

    // std::vector<size_t> shape_;
    // std::vector<double> data_;

    std::vector<expression> exprs_;
    std::map<std::string, std::vector<double>> kwargs_;
    
}; // class evaluation

#endif // EVALUATION_H
