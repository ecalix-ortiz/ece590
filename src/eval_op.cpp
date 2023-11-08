#include "eval_op.h"
#include <assert.h>

/********************* eval_op ***************************/
eval_op::eval_op(const expression &expr):
    expr_id_(expr.expr_id_),
    op_name_(expr.op_name_),
    op_type_(expr.op_type_),
    inputs_(expr.inputs_) {
}
eval_op::~eval_op() {
}
void eval_op::eval(vars_type &variables, const kwargs_type &kwargs) {
    assert(false); // should be provided by derived classes
}
int eval_op::get_expr_id() {
    return expr_id_;
}

/********************* eval_input ***************************/
eval_input::eval_input(const expression &expr):
    eval_op(expr) {
}
void eval_input::eval(vars_type &variables, const kwargs_type &kwargs) {
    variables[expr_id_] = kwargs.find(op_name_)->second;
}

/********************* eval_const ***************************/
eval_const::eval_const(const expression &expr):
    eval_op(expr),
    value_(expr.get_op_param("value")) {
}
void eval_const::eval(vars_type &variables, const kwargs_type &kwargs) {
   variables[expr_id_] = value_;
}

/********************* eval_add ***************************/
eval_add::eval_add(const expression &expr):
    eval_op(expr) {
}
void eval_add::eval(vars_type &variables, const kwargs_type &kwargs) {
    double *in1_ = variables[inputs_[0]].get_data_array();
    double *in2_ = variables[inputs_[1]].get_data_array();
    double sum[variables[inputs_[0]].get_data_size()];
    for(long int i = 0; i < variables[inputs_[0]].get_data_size(); i++) 
        sum[i] = in1_[i]+in2_[i];
    variables[expr_id_] = tensor(variables[inputs_[0]].get_dim(), variables[inputs_[0]].get_shape_array(), sum);
}

/********************* eval_sub ***************************/
eval_sub::eval_sub(const expression &expr):
    eval_op(expr) {
}
void eval_sub::eval(vars_type &variables, const kwargs_type &kwargs) {
    double *in1_ = variables[inputs_[0]].get_data_array();
    double *in2_ = variables[inputs_[1]].get_data_array();
    double sub[variables[inputs_[0]].get_data_size()];
    for(long int i = 0; i < variables[inputs_[0]].get_data_size(); i++) 
        sub[i] = in1_[i]-in2_[i];
    variables[expr_id_] = tensor(variables[inputs_[0]].get_dim(), variables[inputs_[0]].get_shape_array(), sub);
}

/********************* eval_mul ***************************/
eval_mul::eval_mul(const expression &expr):eval_op(expr){}
void eval_mul::eval(vars_type &variables, const kwargs_type &kwargs){
    double *in1_ = variables[inputs_[0]].get_data_array();
    double *in2_ = variables[inputs_[1]].get_data_array();
    // a is scalar, b is tensor
    if (variables[inputs_[0]].get_dim() == 0) {
        double mul[variables[inputs_[1]].get_data_size()] = {0};
        for(long int i = 0; i != variables[inputs_[1]].get_data_size(); ++i)
            mul[i] = in1_[0]*in2_[i];
        variables[expr_id_] = tensor(variables[inputs_[1]].get_dim(), variables[inputs_[1]].get_shape_array(), mul);
    // b is scalar, a is tensor
    } else if (variables[inputs_[1]].get_dim() == 0) {
        double mul[variables[inputs_[0]].get_data_size()] = {0};
        for(long int i = 0; i < variables[inputs_[0]].get_data_size(); ++i)
            mul[i] = in2_[0]*in1_[i];
        variables[expr_id_] = tensor(variables[inputs_[0]].get_dim(), variables[inputs_[0]].get_shape_array(), mul);
    // a and b are tensors
    } else {
        size_t r1 = variables[inputs_[0]].get_shape_array()[0]; //matrix 1 rows
        size_t c1= variables[inputs_[0]].get_shape_array()[1]; //matrix 1 column
        size_t c2 = variables[inputs_[1]].get_shape_array()[1]; //matrix 2 column
        size_t shape[variables[inputs_[0]].get_dim()] = {r1,c2};
        double mul[variables[inputs_[0]].get_shape_array()[0]*variables[inputs_[1]].get_shape_array()[1]] = {0};
        for(size_t i = 0; i < r1; ++i){
            for (size_t j = 0; j < c2; ++j){
                for(size_t k = 0; k < c1; ++k){ 
                    mul[i*c2+j] += in1_[i*c1+k]*in2_[k*c2+j]; 
                }
            }
        }
        variables[expr_id_] = tensor(variables[inputs_[0]].get_dim(), shape, mul);
    }
}