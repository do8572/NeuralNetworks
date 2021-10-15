class Network:
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer.copy())
        
    def feed_forward(self, input):
        temp_input = input;
        
        for layer in self.layers:
            temp_input = layer.feed_forward(temp_input)
            
        return temp_input
    
    def backpropagate(self, error):
        term_error = error
        
        for layer in self.layers[::-1]:
            term_error = layer.backpropagate(term_error);
    
    def train(self, input_array, output_array, n_epochs=10):
        for input, tar in zip(input_array,output_array):
            for i in range(n_epochs):
                pred = self.feed_forward(input)
                print("Epoch: {}".format(i))
                print("Error: {}".format(self.error(tar, pred)))
                print("-----------------------------------------------------------")
                self.backpropagate(self.error_derivate(tar, pred))
            
    def error_derivate(self, tar, pred):
        return tar - pred
            
    def error(self, tar, pred):
        return 0.5 * (tar - pred)*(tar - pred).T