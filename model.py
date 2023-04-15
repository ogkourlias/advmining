#!/usr/bin/env python3

"""
Python module which contains loss functions, activation functions, a Neuron class and a layer class.
To be used with the corresponding assignments.
"""

# METADATA VARIABLES
__author__ = "Orfeas Gkourlias"
__status__ = "WIP"
__version__ = "0.1"

# IMPORTS
import numpy as np
import sys

# Activation functions
def linear(a):
    return a


def sign(a):
    return np.sign(a)


def mean_squared_error(yhat, y):
    return np.square(yhat - y)


def mean_absolute_error(yhat, y):
    return yhat - y


def tanh(a):
    return np.tanh(a)


def derivative(function, delta=0.001):
    def wrapper_derivative(x, *args):
        return (function(x + delta, *args) - function(x - delta, *args)) / (2 * delta)

        wrapper_derivative.__name__ = function.__name__ + '’'
        wrapper_derivative.__qualname__ = function.__qualname__ + '’'

    return wrapper_derivative


# CLASSES
class Neuron():
    def __init__(self, dim, activation=linear, loss=mean_squared_error, bias=0):
        self.dim = dim
        self.activation = activation
        self.loss = loss
        self.bias = bias
        self.weights = [0 for _ in range(dim)]
        self.loss_derivs = []

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

    def partial_fit(self, xs, ys, *, alpha=0.03):
        for xvals, y in zip(xs, ys):
            pre = self.bias + sum(self.weights[i] * xvals[i] for i in range(self.dim))
            yhat = self.activation(pre)
            self.loss_derivs.append(derivative(self.loss)(yhat, y))
            self.bias -= alpha * derivative(self.loss)(yhat, y) * derivative(self.activation)(pre)
            for i in range(self.dim):
                self.weights[i] -= alpha * derivative(self.loss)(yhat, y) * derivative(self.activation)(pre) * xvals[i]

    def fit(self, xs, ys, *, alpha=0.03, epochs=400):
        for epoch in range(epochs):
            self.partial_fit(xs, ys)
            if np.average(self.loss_derivs) <= 0.03:
                self.loss_derivs = []
                break


# MAIN
def main(args):
    """ Main function """
    # FINISH
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
