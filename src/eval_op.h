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
    std::string op_name_, op_type_;
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

#endif // EVAL_OP