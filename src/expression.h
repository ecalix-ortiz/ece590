#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <vector>
#include <string>
#include <map>

class evaluation;

class expression
{
    //evaluation class can access private instance 
    //variables of this (expression) class
    friend class evaluation; 
public:
    //methods
    expression(
        int expr_id,
        const char *op_name,
        const char *op_type,
        int *inputs,
        int num_inputs);

    void add_op_param_double(
        const char *key,
        double value);

    void add_op_param_ndarray(
        const char *key,
        int dim,
        size_t shape[],
        double data[]);
    // public instance variables

private:
    // instance variables
    const int expr_id_;
    std::string op_name_;
    std::string op_type_;
    std::vector<int> inputs_;
    std::map<std::string, double> op_param_;
}; // class expression

#endif // EXPRESSION_H
