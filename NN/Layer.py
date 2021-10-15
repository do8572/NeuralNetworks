import numpy as np

class Layer:
    def __init__(self, dims, activation_unit, learning_rate=0.1, bias=None, name=None, verbose=False):
        self.weights = np.random.uniform(low=0.0, high=1.0, size=dims)
        self.dims=dims
        self.learning_rate = learning_rate
        self.activation_unit = activation_unit
        self.bias = np.random.uniform(low=0.0, high=1.0, size=(1, dims[1]))
        self.name = name
        self.verbose = verbose
    
    def feed_forward(self, x_input):
        self.input = x_input
        prod = np.dot(x_input, self.weights)
        self.output = self.activation_unit.transform(np.dot(x_input, self.weights) + self.bias)
        if(self.verbose):
            print(self.name)
            print("{} * {} = {}".format(x_input, self.weights, prod))
            print(self.activation_unit.transform(prod))
            print("-----------------------------------------------------------")
        return self.output
    
    def backpropagate(self, error):
        #dy = self.activation_unit.derivate_y(self.output)
        term_error = self.activation_unit.term_error(self.output, error)
        #For first layer
        #term_error = error .* derivative of activation function
        #for each weight: grad = term_error .* y 
        grad = np.dot(self.input.T, term_error)
        p_error = np.dot(self.weights, term_error.T).T
        if(self.verbose):
            print(self.name)
            #print("{} * {} = {}".format(error, dy, term_error))
            print("{} * {} = {}".format(self.input.T, term_error, grad))
            print("-----------------------------------------------------------")
        self.weights += self.learning_rate * grad
        self.bias += self.learning_rate * term_error
        return p_error
        
    def copy(self):
        return Layer(self.dims, self.activation_unit, self.learning_rate, self.bias, name=self.name, verbose=self.verbose)