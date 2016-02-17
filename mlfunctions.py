import numpy as np
import numpy.linalg as la
import math

def feedforward_propagation(value_list,weight_matrix,n_hidden_values):
    assert len(value_list) == len(weight_matrix)
    a = []
    for j in range(n_hidden_values):
        aj = 0
        for i in range(len(value_list)):
            aj += weight_matrix[j][i] * value_list[i]
        a.append(aj)
    return a

def sigmoid(aj):
    return 1/(1+math.e**(-aj))

def feedforward_part_two(hidden_values,weight_matrix_two,n_output_layers):
	b = []
	for l in range(n_output_layers):
		bj = 0 
		for j in range(len(hidden_values)):
			bj += weight_matrix_two[j][i]*hidden_values[j]
		b.append(bj)
	return b



    
