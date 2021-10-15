import numpy as np
from Layer import Layer
from ActivationFunction import RELU
from ActivationFunction import Softmax
from Network import Network

verbose = False

layer1 = Layer((2,3), RELU(), name="L1", verbose=verbose)
layer2 = Layer((3,3), RELU(), name="L2", verbose=verbose)
layer3 = Layer((3,3), RELU(), name="L3", verbose=verbose)
layer4 = Layer((3,3), RELU(), name="L4", verbose=verbose)
layer5 = Layer((3,3), Softmax(), name="SS", verbose=verbose)

nn = Network()
nn.add_layer(layer1)
nn.add_layer(layer2)
nn.add_layer(layer3)
nn.add_layer(layer4)
nn.add_layer(layer5)

input = np.matrix([[1,2]])
target = np.matrix([[0,1,0]])
norm = np.linalg.norm(input, axis=1)
input_n = input / norm

nn.train([input_n], [target], n_epochs=100)