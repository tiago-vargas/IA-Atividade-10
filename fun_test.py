from fun import MLP, Perceptron, Neuron
import numpy as np


def sigmoid(x: float) -> float:
	return 1 / (1 + np.exp(-x))


def test_3x2x2():
	network = MLP([3, 2, 2])
	network.layers[0] = [Perceptron(weights=[1], bias=0), Perceptron(weights=[1], bias=0), Perceptron(weights=[1], bias=0)]
	network.layers[1] = [Perceptron(weights=[0.2, -0.1, 0.4], bias=0), Perceptron(weights=[0.7, -1.2, 1.2], bias=0)]
	network.layers[2] = [Perceptron(weights=[1.1, 0.1], bias=0), Perceptron(weights=[3.1, 1.17], bias=0)]
	x = [10.0, 30.0, 20.0]

	y = network.output(x)

	h_0 = sigmoid(x[0] * 0.2 + x[1] * -0.1 + x[2] * 0.4)
	h_1 = sigmoid(x[0] * 0.7 + x[1] * -1.2 + x[2] * 1.2)
	o_0 = sigmoid(h_0 * 1.1 + h_1 * 0.1)
	o_1 = sigmoid(h_0 * 3.1 + h_1 * 1.17)
	assert y[0] == o_0


def test_neuron():
	neuron = Neuron(weights=[0.2, -0.1, 0.4], bias=0)

	sum = neuron.compute_weighted_sum(inputs=[10.0, 30.0, 20.0])

	assert sum == 7


def test_perceptron():
	perceptron = Perceptron(weights=[0.2, -0.1, 0.4], bias=0)

	sum = perceptron.compute_weighted_sum(inputs=[10.0, 30.0, 20.0])

	assert sum == 7
