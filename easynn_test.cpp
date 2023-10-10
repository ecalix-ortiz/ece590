/**
 * A simple test program helps you to debug your easynn implementation.
 */

#include <stdio.h>
#include "src/libeasynn.h"

int main()
{
    program *prog = create_program(); // create blank program and assing pointer of it to prog

    int inputs0[] = {}; // new array of int values
    // adds an expression object to the program prog
    append_expression(
        prog, //program
        0, //expression id
        "a" /*name of expression*/, 
        "Input", //operation type
        inputs0, //array
        0); //number of inputs

    int inputs1[] = {0, 0}; // array of values
    append_expression(prog, 1, "", "Add", inputs1, 2);

    // creating new evaluation pointer from expression in program
    evaluation *eval = build(prog);

    add_kwargs_double(eval, //eval object
        "a", //key of expression
        5); //value

    int dim = 0;
    size_t *shape = nullptr;
    double *data = nullptr;
    // should return a pointer to data 
    if (execute(eval, &dim, &shape, &data) != 0)
    {
        printf("evaluation fails\n");
        return -1;
    }

    if (dim == 0)
        printf("res = %f\n", data[0]);
    else
        printf("result as tensor is not supported yet\n");

    return 0;
}
