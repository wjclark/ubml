import random
import numpy as np
import mlfunctions
import nnScript


n_hidden = 5
data_dim = 5
lambdaval = 2

output_num = 3
dummy_data = [ int( random.random()*10 )%2 for x in range(data_dim) ]  ##768 random 0's or 1's
dummy_data.append(1.0)

dummy_true_values = [ int(random.random()*10 )%2 for x in range(output_num) ]


weights = [ [ random.random()  for x in range(data_dim+1) ]  for x in range(n_hidden) ]
weights = np.array(weights)
print (len(weights), len(weights[0]))


initial_w1 = nnScript.initializeWeights(data_dim, n_hidden);
initial_w2 = nnScript.initializeWeights(n_hidden, output_num);
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

w1 = initialWeights[0:n_hidden * (data_dim + 1)].reshape( (n_hidden, (data_dim + 1)))
w2 = initialWeights[(n_hidden * (data_dim + 1)):].reshape((output_num, (n_hidden + 1)))
# unroll 2 weight matrices into single column vector

#w1 = weights.reshape((n_hidden,data_dim+1))
#w2 = weights.reshape((output_num, n_hidden +1 ))

#for weight in weights:
#	print weight

dummy_data = np.array(dummy_data)



def test_ff_pt1():
	result = mlfunctions.feedforward_propagation(dummy_data,weights, n_hidden )
	print (result)
	return result
	

res1 = test_ff_pt1()
res1 = mlfunctions.matt_sigmoid(res1)
print (res1)

res2 = mlfunctions.feedforward_part_two( res1, w2, output_num )
res2 = mlfunctions.matt_sigmoid(res2)

print( res2)

err_test = mlfunctions.error_function(res2, dummy_true_values, data_dim )
print (err_test)


err_grad_2 = mlfunctions.calc_part_one_grad( dummy_true_values, [res2] )
for item in err_grad_2:
	print (item)


new_weights2 = mlfunctions.backpropagation_hidden_to_output(w2, res2, np.array(err_grad_2).flatten(), lambdaval )

err_grad_1  = mlfunctions.calc_delta_input( res1, err_grad_2, w1 )

print (err_grad_1)

new_weights1 = mlfunctions.backpropagation_hidden_to_output(w1, res1, np.array(err_grad_1) , lambdaval )

print (new_weights1)

calculated_error = 0

calculated_error = mlfunctions.error_function(res2, dummy_true_values, output_num )
print (calculated_error)

print (lambdaval)


predicted = nnScript.nnPredict(new_weights1, new_weights2, dummy_data )
print( predicted )
