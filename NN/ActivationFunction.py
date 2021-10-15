import numpy as np

class ActivationFunction:    
    def transform(self, x):
        pass
    
    def derivate_y(self, y):
        pass
    
    def term_error(self, output, error):
        pass

# RELU
class RELU(ActivationFunction):        
    def transform(self, x):
        return np.maximum(x, 0)
    
    def derivate_y(self, y):
        return (y > 0).astype(float)
    
    def term_error(self, output, error):
        dy = self.derivate_y(output)
        return np.multiply(error, dy)
    
class Softmax(ActivationFunction):    
    def transform(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=1)
    
    def derivate_y(self, y):
        return np.diagflat(y) - np.dot(y.T, y)
    
    def term_error(self, output, error):
        term_error = np.dot(error, self.derivate_y(output))
        return term_error

# TO-DO: Sigmoid
class Sigmoid(ActivationFunction):
    def transform(self, x):
        return 1 / (1 + np.exp(x))
    
    def derivate_y(self, y):
        return y * (1 - y)
    
# Tanh
class Tanh(ActivationFunction):
    def transform(self, x):
        return np.maximum(x, 0)
    
    def derivate_y(self, y):
        return (y > 0).astype(float)
    
    def term_error(self, output, error):
        dy = self.derivate_y(output)
        return np.multiply(error, dy)   
    
class Max(ActivationFunction):    
    def transform(self, x):
        pass
    
    def derivate(self, x):
        pass