import numpy as np
import numpy.linalg as la
import math

def feedforward_propagation(value_list,weight_matrix,n_hidden_values):
    #print len(value_list), len(weight_matrix[0])
    a = []
    
    #print( len(weight_matrix), len(weight_matrix[0]) )
    #assert len(value_list) == len(weight_matrix[0])  ##we are missing the regularization slot now.

    """
    matrix = np.matrix(weight_matrix)
    matrix = matrix[:n_hidden_values,:len(value_list)]  #sub select from the matrix
    
    result = matrix * value_list.reshape((len(value_list),1))
    #print( result )
    """
    for j in range(n_hidden_values):
        aj = 0
        for i in range(len(value_list)):
            aj += weight_matrix[j][i] * value_list[i]
        #print aj
        a.append(aj)
        #print(aj)
    
    #result = result.flatten()
    #int( len(result) == len( a ), result[:5], a[:5] )

    return np.array(a)

def feedforward_part_two(hidden_values,weight_matrix_two,n_output_layers):
    b = []
    print( "gdi", n_output_layers , len(hidden_values) )
    for i in range(n_output_layers):
        bj = 0 
        for j in range(len(hidden_values)):
            #print( i, j )
            #print( weight_matrix_two[j] )
            #print( weight_matrix_two[j][i] )
            #print( hidden_values[j])
            #print( weight_matrix_two[i][j], hidden_values[i] )
            bj += weight_matrix_two[i][j]*hidden_values[i]
        b.append(bj)
    print("B", b)
        #print(bj)
    
    return np.array(b)

def matt_sigmoid( x ):
    print( "sigmoid of ", x)
    print(type(x))
    if type(x) == np.float64:
        val = 1/(1.0+math.e**-x)
        #print va
        return val
    elif type(x) == np.ndarray or type(x) == np.array:
       
        ret_arr =  []
        if type(x[0]) == np.float64: ##if it's an array...
            for num in x:
                ret_arr.append( matt_sigmoid(num) )
            return np.array(ret_arr)

        elif ( type(x[0]) == np.ndarray and type( x[0][0] ) == np.float64 ) : ##if it's a matrix (hopefully)
            new_matrix = []
            for row in x:
                new_matrix.append( matt_sigmoid(row) )
            return np.array(new_matrix)
        else :
            #print (type(x), type(x[0]), type[x[0][0]])
            assert False #somthing has gone really wrong
    else:
        assert False #something has gone suuuper wrong
            

def error_function( values, true_values, n_values ):
    #assert len(values)-1 == n_values 
    #assert len(true_values) == n_values
    sum_of_errors = 0
    #print( len(values), len(true_values) )
    for i in range(len(values)):
        print( len(true_values), len(values), len(true_values), values )
        print (true_values[i], values[i])

        #print(i)
        sum_of_errors += (( true_values[i] - values[i] ) ** 2)
    return .5 * sum_of_errors

def calc_part_one_grad(labels,outputs):
        err_grad2 = []
        for i in range(len(outputs)):
            error_level = []
            for j in range(len(outputs[0])-1):
                print(i,j)
                print( "OUTPUTS", outputs[i][j], labels[i] )
                item_error = calc_delta_hidden(labels[i][j], outputs[i][j])
                error_level.append(item_error)
            err_grad2.append(error_level)
        return err_grad2

def calc_delta_hidden( actual, output ):
    print( "Actual", actual , output)
    assert actual == 0.0 or actual == 1.0
    delta = (actual - output )*output*(1.0-output)
    return delta


def calc_delta_input( outputs, old_deltas, old_weights ):
    print( len(outputs) , len(old_deltas[0]), len(old_weights[0]), len(old_weights) )
    new_deltas = []
    sum_of = 0
    for j in range(len(old_weights)):
        for l in range(len(old_weights[0])):
            #print j,l
            pi_of = old_deltas[j][l] * old_weights[j][l]
            print( "pi", pi_of )
            sum_of += pi_of
        new_delta = outputs[j] * (1-outputs[j] ) * sum_of
        new_deltas.append(new_delta)
    #print( new_deltas )
    return np.array(new_deltas)



def backpropagation_hidden_to_output(weights, outputs, deltas, eta ):
    print( "LEN_DELTAS", len(deltas), "LEN_WEIGHTS", len(weights), len(weights[0]) )
    new_weights = []
    for i in range(len(weights)):
        new_weights_level = []
        for j in range(len(weights[0])):
            new_w = weights[i][j] - eta * deltas[j] * weights[i][j]
            new_weights_level.append( new_w )
        #print( new_weights_level )
        new_weights.append(new_weights_level)
    return np.array(new_weights)





    
if __name__ == "__main__":
    print("commence hand wavey unit tests..." )
    npa = np.array([1.0,2.0])
    for num in npa:
        print( type(num))
    print( matt_sigmoid( npa ))
    npm = np.array( [ [ 1.0, 2.0, 3.0] , [2.2, 1.1, 3.3 ], [4.0,3.0,2.0] ] )
    print( matt_sigmoid( npm ))