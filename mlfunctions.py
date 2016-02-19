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
        #print aj
        a.append(aj)
    return np.array(a)

def feedforward_part_two(hidden_values,weight_matrix_two,n_output_layers):
	b = []
	for l in range(n_output_layers):
		bj = 0 
		for j in range(len(hidden_values)):
			bj += weight_matrix_two[j][i]*hidden_values[j]
		b.append(bj)
	return np.array(b)

def matt_sigmoid(x):
	if type(x) == np.float64:
		val = 1/(1.0+math.e**-x)
		#print val
		return val
	elif type(x) == np.ndarray:
		ret_arr =  []
		if type(x[0]) == np.float64: ##if it's an array...
			for num in x:
				ret_arr.append( matt_sigmoid(num) )
			return np.array(ret_arr)
		elif type(x[0]) == np.ndarray and type( x[0][0] ) == np.float64: ##if it's a matrix (hopefully)
			new_matrix = []
			for row in x:
				new_matrix.append( matt_sigmoid(row) )
			return np.array(new_matrix)
		else :
			assert False #somthing has gone really wrong
	else:
		assert False #something has gone suuuper wrong
			

def error_function( values, true_values, n_values ):
	pass

    
if __name__ == "__main__":
	print "commence hand wavey unit tests..."
	npa = np.array([1.0,2.0])
	for num in npa:
		print type(num)
	print matt_sigmoid( npa )
	npm = np.array( [ [ 1.0, 2.0, 3.0] , [2.2, 1.1, 3.3 ], [4.0,3.0,2.0] ] )
	print matt_sigmoid( npm )