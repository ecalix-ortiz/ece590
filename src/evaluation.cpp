#include <assert.h>
#include "evaluation.h"

evaluation::evaluation(const std::vector<expression> &exprs)
    : result_(0)
{
}

// variables stored here
void evaluation::add_kwargs_double(
    const char *key,
    double value)
{
    // access keyword arguments using key
    kwargs_[key] = value;
    // printing
    printf("value of %s is : %g\n", key, kwargs_[key]);
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
    // printf("value of %s is : %f\n", key, kwargs_["a"]);
    for(auto &expr:exprs_) {
        // bool i1 = expr.op_type_.compare("Input"); // returns 0 if true
        // bool i2 = expr.op_type_.compare("Add");
        // printf("%i %i", i1, i2);
        if(expr.op_type_.compare("Input") == false){
            // expr.add_op_param_double(expr.op_name_.c_str(),expr.inputs_[0]);
            // extract operation name = keyword and
            values_[expr.expr_id_]=kwargs_[expr.op_name_.c_str()];
            // kwargs_[expr.op_name_.c_str()]=expr.inputs_[expr.inputs_.size()-1];
        } else if (expr.op_type_.compare("Const") == false) {
            values_[expr.expr_id_]=expr.op_param_["value"];
        } else if (expr.op_type_.compare("Add") == false) {
            double in1_ = values_[expr.inputs_[0]];
            double in2_ = values_[expr.inputs_[1]];
            values_[expr.expr_id_] = in1_ + in2_;
        } else if (expr.op_type_.compare("Mul") == false) {
            double in1_ = values_[expr.inputs_[0]];
            double in2_ = values_[expr.inputs_[1]];
            values_[expr.expr_id_] = in1_ * in2_;
        } else if (expr.op_type_.compare("Sub") == false) {
            double in1_ = values_[expr.inputs_[0]];
            double in2_ = values_[expr.inputs_[1]];
            values_[expr.expr_id_] = in1_ - in2_;
        }
    }
    result_ = values_[values_.size()-1];
    printf("result: %f\n", result_);
    // for the last expression, get the
    return 0;
}

double &evaluation::get_result()
{
    return result_;
}
