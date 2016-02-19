import numpy as np
import numpy.linalg as la
import math

def feedforward_propagation(value_list,weight_matrix,n_hidden_values):
    #print len(value_list), len(weight_matrix[0])-1
    assert len(value_list) == len(weight_matrix[0])-1  ##we are missing the regularization slot now.
    a = []
    for j in range(n_hidden_values):
        aj = 0
        for i in range(len(value_list)):
            aj += weight_matrix[j][i] * value_list[i]
        a.append(aj)
    return a


def feedforward_part_two(hidden_values,weight_matrix_two,n_output_layers):
	b = []
	for l in range(n_output_layers):
		bj = 0 
		for j in range(len(hidden_values)):
			bj += weight_matrix_two[j][i]*hidden_values[j]
		b.append(bj)
	return b

def error_function( values, true_values, n_values ):
	pass

    
