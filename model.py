import math
import numpy as np


def linear(a):
    return a


def sign(a):
    return np.sign(a)


def mean_squared_error(yhat, y):
    return np.square(yhat - y)


def mean_absolute_error(yhat, y):
    return yhat - y


def derivative(function, delta=0.001):
    def wrapper_derivative(x, *args):
        return (function(x + delta, *args) - function(x - delta, *args)) / (2 * delta)

        wrapper_derivative.__name__ = function.__name__ + '’'
        wrapper_derivative.__qualname__ = function.__qualname__ + '’'

    return wrapper_derivative


class Neuron():
    def __init__(self, dim, activation = linear, loss = mean_squared_error, bias=0):
        self.dim = dim
        self.activation = activation
        self.loss = loss
        self.bias = bias
        self.weights = [0 for _ in range(dim)]

    def __repr__(self):
        text = f'Neuron(dim={self.dim}, activation={self.activation.__name__}, loss={self.loss.__name__})'
        return text

    def predict(self, xs):
        predicted = []
        for xvals in xs:
            pre = self.bias + sum(self.weights[i] * xvals[i] for i in range(self.dim))
            post = self.activation(pre)
            predicted.append(post)
        return predicted

    def partial_fit(self, xs, ys, *, alpha = 0.03):
        for xvals, y in zip(xs, ys):
            pre = self.bias + sum(self.weights[i] * xvals[i] for i in range(self.dim))
            yhat = self.activation(pre)
            self.bias -= alpha * derivative(self.loss)(yhat,y) * derivative(self.activation)(pre)
            for i in range(self.dim):
                self.weights[i] -= alpha * derivative(self.loss)(yhat,y) * derivative(self.activation)(pre) * xvals[i]

    def fit(self,  xs, ys, *, alpha= 0.03, epochs= 5):
        for epoch in range(epochs):
            self.partial_fit(xs, ys)

    def tanh(self, a):
        return np.e