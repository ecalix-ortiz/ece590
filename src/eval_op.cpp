#include "eval_op.h"
#include <assert.h>
#include <limits>
#include <algorithm>

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
eval_mul::eval_mul(const expression &expr):
    eval_op(expr) {
}
void eval_mul::eval(vars_type &variables, const kwargs_type &kwargs) {
    double *in1_ = variables[inputs_[0]].get_data_array();
    double *in2_ = variables[inputs_[1]].get_data_array();
    // a is scalar, b is tensor
    if (variables[inputs_[0]].get_dim() == 0) {
        double mul[variables[inputs_[1]].get_data_size()] = {0};
        for (long int i = 0; i != variables[inputs_[1]].get_data_size(); ++i)
            mul[i] = in1_[0]*in2_[i];
        variables[expr_id_] = tensor(variables[inputs_[1]].get_dim(), variables[inputs_[1]].get_shape_array(), mul);
    // b is scalar, a is tensor
    } else if (variables[inputs_[1]].get_dim() == 0) {
        double mul[variables[inputs_[0]].get_data_size()] = {0};
        for (long int i = 0; i < variables[inputs_[0]].get_data_size(); ++i)
            mul[i] = in2_[0]*in1_[i];
        variables[expr_id_] = tensor(variables[inputs_[0]].get_dim(), variables[inputs_[0]].get_shape_array(), mul);
    // a and b are tensors
    } else {
        // get matrix columns and rows for iteration
        size_t r1 = variables[inputs_[0]].get_shape_array()[0];
        size_t c1 = variables[inputs_[0]].get_shape_array()[1];
        size_t c2 = variables[inputs_[1]].get_shape_array()[1];
        // use the dimensions to get shape
        size_t shape[variables[inputs_[0]].get_dim()] = {r1,c2};
        double mul[variables[inputs_[0]].get_shape_array()[0]*variables[inputs_[1]].get_shape_array()[1]] = {0};
        // iterate through the row and columns of the matrices
        for(size_t i = 0; i < r1; ++i)
            for (size_t j = 0; j < c2; ++j)
                for(size_t k = 0; k < c1; ++k)
                    mul[i*c2+j] += in1_[i*c1+k]*in2_[k*c2+j];
        // save result as tensor on variables
        variables[expr_id_] = tensor(variables[inputs_[0]].get_dim(), shape, mul);
    }
}

/********************** eval_relu *****************************/
eval_relu::eval_relu(const expression &expr):
    eval_op(expr) {
}
void eval_relu::eval(vars_type &variables, const kwargs_type &kwargs) {
    // apply the ReLU operation element-wise to the elements in the input tensor
    std::vector<double> relu;
    for(long int i = 0; i < variables[inputs_[0]].get_data_size(); ++i){
        if(variables[inputs_[0]].get_data_array()[i] < 0)
            relu.push_back(0);
        else
            relu.push_back(variables[inputs_[0]].get_data_array()[i]);
    }
    variables[expr_id_] = tensor(variables[inputs_[0]].get_dim(), variables[inputs_[0]].get_shape_array(), &relu[0]);
}

/********************** eval_flatten *****************************/
eval_flatten::eval_flatten(const expression &expr):
    eval_op(expr) {
}
void eval_flatten::eval(vars_type &variables, const kwargs_type &kwargs) {
    // flattening each example into a vector using row-major order
    // shape[0]=N
    size_t *N = variables[inputs_[0]].get_shape_array();
    // C H W params
    size_t C = N[1];
    size_t H = N[2];
    size_t W = N[3];
    // shape[1]=C*H*W
    size_t flatten_shape[2] = {N[0], C*H*W};
    variables[expr_id_] = tensor(2, flatten_shape, variables[inputs_[0]].get_data_array());
}

/********************** eval_input2d *****************************/
eval_input2d::eval_input2d(const expression &expr):
    eval_op(expr),
    height_(expr.get_op_param("height")),
    width_(expr.get_op_param("width")),
    in_channels_(expr.get_op_param("in_channels")) {
}
void eval_input2d::eval(vars_type &variables, const kwargs_type &kwargs) {
    /* obtain the input tensor in NHWC format using name and output it in NCHW format. The parameters 
    height, width, and in_channels allow you to verify the shapes of the input tensor, though you don’t 
    have to check for them. */
    // find the item that is the op_name_ without it being the last item of kwargs
    auto iter = kwargs.find(op_name_);
    assert(iter != kwargs.end());
    tensor input2d_ = iter->second;

    const auto& shape = input2d_.get_shape_array();
    // const auto& data = input2d_.get_data_array();

    // confirm that the height, width, and in_channels match that of the shape
    assert(input2d_.get_dim() == 4);
    assert(shape[1] == height_.item());
    assert(shape[2] == width_.item());
    assert(shape[3] == in_channels_.item());

    // instantiate the variables
    const size_t N = shape[0];
    const size_t H = shape[1];
    const size_t W = shape[2];
    const size_t C = shape[3];

    // final shape and data arrays that will be passed to the new tensor
    size_t s[4] = {N,C,H,W};
    double d[input2d_.get_data_size()] = {0};

    // perform search
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    d[(n*C*H*W)+(c*H*W)+(h*W)+w] = input2d_.at(n, h, w, c);
                }
            }
        }
    }
    variables[expr_id_] = tensor(4, s, d);
}

/********************** eval_linear *****************************/
eval_linear::eval_linear(const expression &expr):
    eval_op(expr),
    weight_(expr.get_op_param("weight")),
    bias_(expr.get_op_param("bias")) {
}
void eval_linear::eval(vars_type &variables, const kwargs_type &kwargs) {
    /* the input tensor is a matrix of N rows and I columns, weight is a
    matrix with O rows and I columns, bias is a vector of O elements, and the output tensor is a 
    matrix of N rows and O columns. For each row x of the input tensor, compute a row in the output 
    tensor as (weight x⊤ + bias)⊤. */
    size_t N = variables[inputs_[0]].get_shape_array()[0];
    size_t O = weight_.get_shape_array()[0];
    size_t I = weight_.get_shape_array()[1];

    assert(variables[inputs_[0]].get_shape_array()[1] == I);
    assert(bias_.get_shape_array()[0] == O);

    size_t s[2] = {N, O};

    double temp = 0;
    std::vector<double> d;
    // perform (weight x⊤ + bias)⊤ for each row in the output tensor
    for (size_t n = 0; n < N; ++n) {
        for (size_t o = 0; o < O; ++o) {
            for (size_t i = 0; i < I; ++i) 
                temp += weight_.at(o,i) * variables[inputs_[0]].at(n,i);
            d.push_back(temp+bias_.at(o));
            temp = 0;
        }
    }
    variables[expr_id_] = tensor(2, s, &d[0]);
}

/********************** eval_maxpool2d *****************************/
eval_maxpool2d::eval_maxpool2d(const expression &expr):
    eval_op(expr),
    kernel_size(expr.get_op_param("kernel_size").get_data_array()[0]),
    stride(expr.get_op_param("stride").get_data_array()[0]) {
}
void eval_maxpool2d::eval(vars_type &variables, const kwargs_type &kwargs){
    /* the input tensor is a 4D tensor with the shape (N, C, H, W). The parameters kernel_size 
    and stride are both integers (you will need to extract them from the parameter tensors and 
    convert them to integers). For simplicity, assume kernel_size = stride. For each 
    kernel_size * kernel_size patch from one slice per example per channel from the input tensor, 
    their maximum is put into the output tensor. Since we assume kernel_size = stride, patches 
    won’t overlap and partial patches at the boundary will be discarded.

    Overall, the output should be a 4D tensor with the shape (N, C, H/kernel_size,
    W/kernel_size). */
    assert(variables[inputs_[0]].get_dim() == 4);

    auto& input_tensor = variables[inputs_[0]];
    const auto& input_shape = variables[inputs_[0]].get_shape_array();

    size_t N = input_shape[0];
    size_t C = input_shape[1];
    size_t H = input_shape[2];
    size_t W = input_shape[3];

    size_t output_H = H / kernel_size;
    size_t output_W = W / kernel_size;

    size_t output_shape[4] = {N, C, output_H, output_W}; // output array
    std::vector<double> output_data;

    // Iterate over input patches
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            // Extract slice for each channel
            std::vector<double> slice;
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    slice.push_back(input_tensor.at(n, c, h, w));
                }
            }
            // Max pooling within each patch
            for (size_t h = 0; h < output_H; ++h) {
                for (size_t w = 0; w < output_W; ++w) {
                    std::vector<double> patch_values;
                    for (int r = 0; r < kernel_size; ++r) {
                        for (int c = 0; c < kernel_size; ++c) {
                            patch_values.push_back(slice[(r * W + c + w * kernel_size) + (h * kernel_size * W)]);
                        }
                    }
                    auto max_val = *std::max_element(patch_values.begin(), patch_values.end());
                    output_data.push_back(max_val);
                }
            }
        }
    }
    variables[expr_id_] = tensor(4, output_shape, &output_data[0]);
}

/********************** eval_maxpool2d *****************************/
eval_conv2d::eval_conv2d(const expression &expr):
    eval_op(expr),
    weight_(expr.get_op_param("weight")),
    bias_(expr.get_op_param("bias")),
    in_channels_(size_t(expr.get_op_param("in_channels").get_data_array()[0])),
    out_channels_(size_t(expr.get_op_param("out_channels").get_data_array()[0])),
    kernel_size_(size_t(expr.get_op_param("kernel_size").get_data_array()[0])) {
}
void eval_conv2d::eval(vars_type &variables, const kwargs_type &kwargs){
    /* the input tensor is a 4D
    tensor with the shape (N, in_channels, H, W), weight is a 4D tensor with the shape (out_channels, 
    in_channels, kernel_size, kernel_size), bias is a vector with out_channels elements. (Note that 
    the parameters in_channels, out_channels, and kernel_size allow you to verify the shapes of the 
    input tensor, weight, and bias, though you don’t have to check for them.) For each 
    in_channels * kernel_size * kernel_size tensor from one slice per example from the input tensor, 
    it is multiplied element-wise with one slice per out_channels from weight, and then the result 
    elements, plus the corresponding element from bias for out_channels, are added together and put 
    into the output tensor. Overall, the output should be a 4D tensor with the shape (N, out_channels, 
    H-kernel_size+1, W-kernel_size+1). */
    // Extracting shapes of input and weights
    auto& input_tensor = variables[inputs_[0]];
    const auto& input_shape = variables[inputs_[0]].get_shape_array();

    size_t N = input_shape[0];
    size_t I = input_shape[1];
    size_t H = input_shape[2];
    size_t W = input_shape[3];

    // const auto& weight_shape = weight_.get_shape_array();

    // for debugging
    // assert(weight_shape[0] == out_channels_);
    // assert(weight_shape[1] == in_channels_);
    // assert(weight_shape[2] == kernel_size_);
    // assert(weight_shape[3] == kernel_size_);

    // Calculating output shape
    size_t output_shape[4] = {N, out_channels_, (H - kernel_size_ + 1), (W - kernel_size_ + 1)};

    std::vector<double> output_data;
    std::vector<double> input_slice;

    // Loop through each input sample
    for (size_t n = 0; n < N; ++n) {
        input_slice.clear();
        // Collect slice from input tensor
        for (size_t i = 0; i < I; ++i)
            for (size_t h = 0; h < H; ++h)
                for (size_t w = 0; w < W; ++w)
                    input_slice.push_back(input_tensor.at(n, i, h, w));

        std::vector<double> weight_input_slice;
        std::vector<double> input_tensor_input_slice;

        // Loop through each output channel
        for (size_t o = 0; o < out_channels_; ++o) {
            std::vector<double> output_convolution;

            // Convolution operation
            for (size_t i = 0; i < I; ++i) {
                weight_input_slice.clear();
                // Collect slice from weights for current input and output channel
                for (size_t kh = 0; kh < kernel_size_; ++kh)
                    for (size_t kw = 0; kw < kernel_size_; ++kw)
                        weight_input_slice.push_back(weight_.at(o, i, kh, kw));

                input_tensor_input_slice.clear();
                // Collect slice from input tensor for the current input channel
                for (size_t h = 0; h < H; ++h)
                    for (size_t w = 0; w < W; ++w)
                        input_tensor_input_slice.push_back(input_slice[(i * H * W) + (h * W) + w]);

                std::vector<double> mul;

                // Convolution operation within the input channels
                size_t I_position_index = 0; // calculate once per i
                size_t W_position_index = 0; // calculate once per i
                for (size_t h = 0; h <= (H - kernel_size_); ++h) {
                    for (size_t w = 0; w <= (W - kernel_size_); ++w) {
                        double sum = 0;
                        //Element-wise multiplication and accumulation
                        for (size_t r = 0; r < kernel_size_; ++r) {
                            for (size_t c = 0; c < kernel_size_; ++c) {
                                size_t I_pos = I_position_index + (h * W) + w + (r * W) + c;
                                size_t W_pos = W_position_index + (r * kernel_size_) + c;
                                sum += input_tensor_input_slice[I_pos] * weight_input_slice[W_pos];
                            }
                        }
                        mul.push_back(sum);
                    }
                }

                // Applying bias and summing results
                if (i == 0)
                    for (size_t p = 0; p < mul.size(); ++p)
                        output_convolution.push_back(mul[p] + bias_.at(o));
                else
                    for (size_t p = 0; p < mul.size(); ++p)
                        output_convolution[p] += mul[p];
            }

            // Adding results to the final vector
            for (size_t p = 0; p < output_convolution.size(); ++p) {
                output_data.push_back(output_convolution[p]);
            }
        }
    }

    // Storing the result in the variables under expr_id_
    variables[expr_id_] = tensor(4, output_shape, &output_data[0]);
}