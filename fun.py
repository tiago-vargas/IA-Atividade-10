from typing import Callable, TypeVar
import numpy as np


def sigmoid(x: float) -> float:
	return 1 / (1 + np.exp(-x))


class Neuron:
	def __init__(self, weights: list[float], bias:float):
		self.weights = weights
		self.bias = bias

	T = TypeVar('T')

	def classify(self, inputs: list[float], classifier: Callable[[float], T]) -> T:
		y = self.compute_weighted_sum(inputs)

		return classifier(y)

	def compute_weighted_sum(self, inputs: list[float]) -> float:
		z = zip(inputs, self.weights)
		products = []
		for (x, w) in z:
			products.append(x * w)

		y = sum(products) + self.bias
		return y


class Perceptron(Neuron):
	def __init__(self, n: int=0, weights: list[float]=[], bias: float=0):
		if n == 0:
			super().__init__(weights, bias)
		else:
			w = np.random.randn(n)
			b = np.random.randn(1)
			super().__init__(weights=list(w), bias=float(b))

	def train(self, inputs: list[list[float]], target: list[float], alpha: float, precision: float, max_iter=100):
		def output(inputs: list[float]) -> float:
			z = zip(inputs, self.weights)
			products = []
			for (x, w) in z:
				products.append(x * w)
			y = sum(products) + self.bias

			return y

		actual = [output(inputs[i]) for i in range(len(inputs))]
		i = 0
		error = np.subtract(target, actual)
		while i < max_iter and np.abs(error).max() >= precision:
			for (x, expected) in zip(inputs, target):
				actual = output(x)
				delta = expected - actual
				np.add(self.weights, [alpha * delta * x[i] for i in range(len(x))])
				self.bias += alpha * delta
			actual = [output(inputs[i]) for i in range(len(inputs))]
			error = np.subtract(target, output(actual))
			i += 1


class MLP:
	def __init__(self, layer_sizes: list[int]):
		self.layers: list[list[Perceptron]] = []

		assert len(layer_sizes) >= 3, 'MLP must have at least 3 layers'

		input_size = layer_sizes[0]
		self.layers.append([Perceptron(1) for _ in range(input_size)])

		for i in range(len(layer_sizes[1:])):
			previous_layer_size = layer_sizes[i - 1]
			current_layer_size = layer_sizes[i]
			self.layers.append([Perceptron(previous_layer_size) for _ in range(current_layer_size)])

	def output(self, inputs: list[float]) -> list[float]:
		# A primeira camada não calcula nada
		layer_outputs = inputs

		for i in range(1, len(self.layers)):
			layer = self.layers[i]
			layer_inputs = layer_outputs
			layer_outputs: list[float] = []
			for neuron in layer:
				xwb = neuron.compute_weighted_sum(inputs=layer_inputs)
				y = sigmoid(xwb)
				layer_outputs.append(y)

		return layer_outputs

	def classify(self, inputs: list[float]) -> int:
		y = self.output(inputs)
		max = np.max(y)
		return round(max)
