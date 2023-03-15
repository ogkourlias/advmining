import math

import numpy


class Perceptron():
    def __init__(self, dim):
        self.dim = dim
        self.bias = 0
        self.weights = [0, 0]

    def __repr__(self):
        text = f'Perceptron(dim={self.dim})'
        return text

    def predict(self, xs):
        final_list = []
        for x in xs:
            summed = 0
            for point in range(self.dim):
                summed += x[point] * self.weights[point]
            final_list.append(numpy.sign(self.bias + summed))
        return final_list

    def partial_fit(self, xs, ys):
        for x, y in zip(xs, ys):
            summed = 0
            for point in range(self.dim):
                summed += x[point] * self.weights[point]
                self.weights[point] -= (numpy.sign(self.bias + summed) - y) * x[point]
            self.bias -= (numpy.sign(self.bias + summed) - y)
            # Update hier het perceptron met één instance {x, y}

    def fit(self, xs, ys, *, epochs=0):
        if epochs != 0:
            for epoch in range(epochs):
                old_bias = self.bias
                old_wght = self.weights
                self.partial_fit(xs, ys)
                if old_bias == self.bias and old_wght == self.weights:
                    break
        else:
            go = 1
            while go == 1:
                old_bias = self.bias
                old_wght = self.weights
                self.partial_fit(xs, ys)
                if old_bias == self.bias and old_wght == self.weights:
                    go = 0


class LinearRegression():
    def __init__(self, dim):
        self.dim = dim
        self.bias = 0
        self.weights = [0, 0]

    def __repr__(self):
        text = f'Perceptron(dim={self.dim})'
        return text

    def predict(self, xs):
        final_list = []
        for x in xs:
            summed = 0
            for point in range(self.dim):
                summed += x[point] * self.weights[point]
            final_list.append(self.bias + summed)
        return final_list

    def partial_fit(self, xs, ys, alpha=0.01):
        for x, y in zip(xs, ys):
            y_roof = 0
            for dim in range(self.dim):
                y_roof += self.bias + x[dim] * self.weights[dim]
                self.weights[dim] -= alpha * (y_roof - y) * x[dim]
            print(self.bias)
            self.bias -= alpha * (y_roof - y)
            # Update hier het perceptron met één instance {x, y}

    def fit(self, xs, ys, *, epochs=1000, alpha=0.01):
        if epochs != 0:
            for epoch in range(epochs):
                old_bias = self.bias
                old_wght = self.weights
                self.partial_fit(xs, ys)
                wgth_diff = []
                for i, j in zip(old_wght, self.weights):
                    wgth_diff.append(min(i, j) / max(i, j))
                if 0.025 * old_bias <= abs(old_bias - self.bias) and max(wgth_diff) <= 1 - alpha / 2:
                    break

### Activation functions ###

def linear(a):
    return a

def sign(a):
    if a > 0:
        return 1
    if a < 0:
        return -1
    else:
        return 0

def tanh(a):
    return math.tanh(a)

### Loss functions ###

def mean_squared_error(yhat, y):
    return math.pow(yhat - y, 2)

def mean_absolute_error(yhat, y):
    return abs(yhat - y)

def hinge(yhat, y):
    return max(1 - yhat * y, 0)

def derivative(function, delta=0.01):
    def wrapper_derivative(x, *args):
        return (function(x + delta, *args) - function(x - delta, *args)) / (2 * delta)

    wrapper_derivative.__name__ = function.__name__ + '’'
    wrapper_derivative.__qualname__ = function.__qualname__ + '’'
    return wrapper_derivative

class Neuron:
    def __init__(self, dim=2, activation=linear, loss=mean_squared_error):
        self.dim = dim
        self.bias = 0.0
        self.weights = [0.0] * dim
        self.activation = activation
        self.loss = loss

    def __repr__(self):
        text = f"Neuron(dim={self.dim}, activation={self.activation.__name__}, loss={self.loss.__name__})"
        return text

    def predict(self, xs):
        yhats = []

        for x in xs:
            pre_activation = self.bias + sum(self.weights[i] * x[i] for i in range(self.dim))
            post_activation = self.activation(pre_activation)
            yhats.append(post_activation)

        return yhats

    def partial_fit(self, xs, ys, *, alpha = 0.01):
        yhats = self.predict(xs)

        for x, y, yhat in zip(xs, ys, yhats):
            self.bias = self.bias - alpha * derivative(self.loss)(yhat, y) * derivative(self.activation)(
                self.bias + sum(self.weights[i] * x[i] for i in range(self.dim))
            )

            self.weights = [self.weights[i] - alpha * derivative(self.loss)(yhat, y) * derivative(self.activation)(
                self.bias + sum(self.weights[i] * x[i] for i in range(self.dim))
            )
                             * x[i] for i in range(self.dim)]

    def fit(self, xs, ys, *, alpha = 0.001, epochs=1000):
        epochs_done = 0
        while epochs_done < epochs:
            self.partial_fit(xs, ys, alpha=alpha)
            epochs_done += 1