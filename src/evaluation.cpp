#include <assert.h>
#include "evaluation.h"

evaluation::evaluation(const std::vector<expression> &exprs) {
    for (auto &expr: exprs) {
        if (expr.op_type_.compare("Input") == false)
            ops_.push_back(std::make_shared<eval_input>(expr));
        else if (expr.op_type_.compare("Const") == false)
            ops_.push_back(std::make_shared<eval_const>(expr));
        else if (expr.op_type_.compare("Add") == false)
            ops_.push_back(std::make_shared<eval_add>(expr));
        else if (expr.op_type_.compare("Sub") == false)
            ops_.push_back(std::make_shared<eval_sub>(expr));
        else if (expr.op_type_.compare("Mul") == false)
            ops_.push_back(std::make_shared<eval_mul>(expr));
        else if (expr.op_type_.compare("ReLU") == false)
            ops_.push_back(std::make_shared<eval_relu>(expr));
        else if (expr.op_type_.compare("Flatten") == false)
            ops_.push_back(std::make_shared<eval_flatten>(expr));
        else if (expr.op_type_.compare("Input2d") == false)
            ops_.push_back(std::make_shared<eval_input2d>(expr));
        else if (expr.op_type_.compare("Linear") == false)
            ops_.push_back(std::make_shared<eval_linear>(expr));
        else if (expr.op_type_.compare("MaxPool2d") == false)
            ops_.push_back(std::make_shared<eval_maxpool2d>(expr));
        else if (expr.op_type_.compare("Conv2d") == false)
            ops_.push_back(std::make_shared<eval_conv2d>(expr));
    }
}

// variables stored here
void evaluation::add_kwargs_double(
    const char *key,
    double value)
{
    // access keyword arguments using key
    kwargs_[key] = tensor(value);
    // printing
    // printf("value of %s is : %g\n", key, kwargs_[key]);
}

void evaluation::add_kwargs_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    kwargs_[key] = tensor(dim, shape, data);
}

int evaluation::execute()
{
    /*
    variables_.clear();
    for(auto &expr:exprs_) {
        if(expr.op_type_.compare("Input") == false){
            variables_[expr.expr_id_] = kwargs_[expr.op_name_.c_str()];
            // variables_[expr.expr_id_] = kwargs.find(expr.op_name_.c_str());

        } else if (expr.op_type_.compare("Const") == false) {
            variables_[expr.expr_id_] = expr.op_param_["value"];

        } else if (expr.op_type_.compare("Add") == false) {
            double *in1_ = variables_[expr.inputs_[0]].get_data_array();
            double *in2_ = variables_[expr.inputs_[1]].get_data_array();
            double sum[variables_[expr.inputs_[0]].get_dim()];
            for(int i = 0; i < variables_[expr.inputs_[0]].item(); i++) 
                sum[i] = in1_[i]+in2_[i];
            variables_[expr.expr_id_] = tensor(variables_[expr.inputs_[0]].get_dim(), variables_[expr.inputs_[0]].get_shape_array(), sum);

        } else if (expr.op_type_.compare("Sub") == false) {
            std::vector<double> in1_ = variables_[expr.inputs_[0]];
            std::vector<double> in2_ = variables_[expr.inputs_[1]];
            double sub[variables_[expr.inputs_[0]].size()];
            for(long unsigned int i = 0; i < variables_[expr.inputs_[0]].size(); i++) 
                sub[i] = in1_[i]-in2_[i];
            int N=1;
            for(long unsigned int x=0; x<expr.shape_.size();x++) N=N*expr.shape_[x];
            variables_[expr.expr_id_].assign(sub, sub+N);

        } else if (expr.op_type_.compare("Mul") == false) {
            std::vector<double> in1_ = variables_[expr.inputs_[0]];
            std::vector<double> in2_ = variables_[expr.inputs_[1]];

            // a is scalar, b is tensor
            if (variables_[expr.inputs_[0]].size() == 0) {
                double mul[variables_[expr.inputs_[1]].size()];
                for(long unsigned int i = 0; i != variables_[expr.inputs_[1]].size(); i++) 
                    mul[i] = in1_[i]*in2_[i];
                int N=1;
                for(long unsigned int x=0; x<expr.shape_.size();x++) N=N*expr.shape_[x];
                variables_[expr.expr_id_].assign(mul, mul+N);
            // b is scalar, a is tensor
            } else if (variables_[expr.inputs_[1]].size() == 0) {
                double mul[variables_[expr.inputs_[0]].size()];
                for(long unsigned int i = 0; i != variables_[expr.inputs_[0]].size(); i++) 
                    mul[i] = in1_[i]*in2_[i];
                int N=1;
                for(long unsigned int x=0; x<expr.shape_.size();x++) N=N*expr.shape_[x];
                variables_[expr.expr_id_].assign(mul, mul+N);
            // a and b are tensors
            // row size of a and col size of b is used for final size matrix
            } else {
                double mul[variables_[expr.inputs_[0]].size()*variables_[expr.inputs_[1]].size()] = {0};
                // take the row of matrix 1 and col of matrix 2
                size_t row = variables_[expr.inputs_[0]].size();
                size_t col = variables_[expr.inputs_[1]].size();
                size_t shape[variables_[expr.inputs_[0]].size()] = {row,col};
                for (size_t i = 0; i < row; i++) {
                    for (size_t j = 0; j < col; j++) {
                        mul[i*col+j] += in1_[i*col+j]*in2_[j*col+j];
                    }
                }
                variables_[expr.expr_id_].assign(shape, mul);
            }
        }
    }
    result_ = variables_[exprs_[exprs_.size()-1].expr_id_];
    return 0;
    */

    // NEW IMPLEMENTATION:
    variables_.clear();
    for (auto &op: ops_) {
        op->eval(variables_, kwargs_);
    }
    result_ = variables_[ops_[ops_.size()-1]->get_expr_id()];
    return 0;
}

tensor &evaluation::get_result()
{
    return result_;
}
