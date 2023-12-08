#ifndef EVAP_OP_H
#define EVAL_OP_H

#include "tensor.h"
#include "map"
#include "vector"
#include "expression.h"

typedef std::map<int, tensor> vars_type;
typedef std::map<std::string, tensor> kwargs_type;

/********************* eval_op ***************************/
class eval_op {
protected:
    int expr_id_;
    std::string op_name_,
                op_type_;
    std::vector<int> inputs_;
public:
    eval_op(const expression &expr);
    virtual ~eval_op();
    virtual void eval(vars_type &variables, const kwargs_type &kwargs) = 0;
    int get_expr_id();
}; // class eval_op

/********************* eval_input ******************************/
class eval_input: public eval_op {
public:
    eval_input(const expression &expr);
    void eval(vars_type &variables , const kwargs_type &kwargs) override;
}; // class eval_input

/********************* eval_const ******************************/
class eval_const: public eval_op {
    tensor value_;
public:
    eval_const(const expression &expr);
    void eval(vars_type &variables , const kwargs_type &kwargs) override;
}; // class eval_const

/********************* eval_add ******************************/
class eval_add: public eval_op {
public:
    eval_add(const expression &expr);
    void eval(vars_type &variables , const kwargs_type &kwargs) override;
};

/********************** eval_sub *****************************/
class eval_sub: public eval_op{
public:
    eval_sub(const expression &expr);
    void eval(vars_type &variables, const kwargs_type &kwargs) override;
};

/********************** eval_mul *****************************/
class eval_mul: public eval_op{
public:
    eval_mul(const expression &expr);
    void eval(vars_type &variables, const kwargs_type &kwargs) override;
};

/********************** eval_relu *****************************/
class eval_relu: public eval_op{
public:
    eval_relu(const expression &expr);
    void eval(vars_type &variables, const kwargs_type &kwargs) override;
};

/********************** eval_flatten *****************************/
class eval_flatten: public eval_op {
public:
    eval_flatten(const expression &expr);
    void eval(vars_type &variables, const kwargs_type &kwargs) override;
};

/********************** eval_input2d *****************************/
class eval_input2d: public eval_op {
private:
    tensor height_, 
           width_, 
           in_channels_;
public:
    eval_input2d(const expression &expr);
    void eval(vars_type &variables, const kwargs_type &kwargs) override;
};

/********************** eval_linear *****************************/
class eval_linear: public eval_op {
private:
    tensor weight_, 
           bias_;
public:
    eval_linear(const expression &expr);
    void eval(vars_type &variables, const kwargs_type &kwargs) override;
};

/********************** eval_maxpool2d *****************************/
class eval_maxpool2d: public eval_op {
private:
    int kernel_size, 
        stride;
public:
    eval_maxpool2d(const expression &expr);
    void eval(vars_type &variables, const kwargs_type &kwargs) override;
};

/********************** eval_conv2d *****************************/
class eval_conv2d: public eval_op {
private:
    tensor weight_, 
           bias_;
    size_t in_channels_, 
           out_channels_, 
           kernel_size_;
public:
    eval_conv2d(const expression &expr);
    void eval(vars_type &variables, const kwargs_type &kwargs) override;
};

#endif // EVAL_OP