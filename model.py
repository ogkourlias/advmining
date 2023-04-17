#!/usr/bin/env python3

"""
Python module which contains loss functions, activation functions, a Neuron class and a layer class.
To be used with the corresponding assignments.
"""

# METADATA VARIABLES
__author__ = "Orfeas Gkourlias"
__status__ = "WIP"
__version__ = "0.1"

import random
# IMPORTS
import sys
import numpy as np
from copy import deepcopy
from collections import Counter

from pandas import DataFrame

import data


# Activation functions
def linear(a):
    return a


def sign(a):
    return np.sign(a)


# Loss functions
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
class Neuron:
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

    def fit(self, xs, ys, *, alpha=0.03, epochs=4000):
        for epoch in range(epochs):
            self.partial_fit(xs, ys)
            if np.average(self.loss_derivs) <= 0.03:
                self.loss_derivs = []
                break


class Layer:
    """A layer of neurons."""
    classcounter = Counter()

    def __init__(self, outputs, *, name=None, next=None):
        """Initialise a layer with a given amount of outputs and a name."""
        Layer.classcounter[type(self)] += 1
        if name is None:
            name = f'{type(self).__name__}_{Layer.classcounter[type(self)]}'
        self.inputs = 0
        self.outputs = outputs
        self.name = name
        self.next = next

    def __add__(self, next):
        """Add a layer to the current layer. (Dunder method)"""
        result = deepcopy(self)
        result.add(deepcopy(next))
        return result

    def __getitem__(self, index):
        if index == 0 or index == self.name:
            return self
        if isinstance(index, int):
            if self.next is None:
                raise IndexError('Layer index out of range')
            return self.next[index - 1]
        if isinstance(index, str):
            if self.next is None:
                raise KeyError(index)
            return self.next[index]
        raise TypeError(f'Layer indices must be integers or strings, not {type(index).__name__}')

    def __call__(self, xs):
        raise NotImplementedError('Abstract __call__ method')

    def __repr__(self):
        """Return a string representation of the layer."""
        text = f'Layer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def add(self, next):
        """Add a layer to the current layer."""
        if self.next is None:
            self.next = next
            next.set_inputs(self.outputs)
        else:
            self.next.add(next)

    def set_inputs(self, inputs):
        self.inputs = inputs


class InputLayer(Layer):
    """Input layer of a neural network."""

    def __call__(self, xs, ys=None, alpha=None):
        return self.next(xs, ys, alpha)

    def __repr__(self):
        text = f'InputLayer(outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def set_inputs(self, inputs):
        raise AttributeError('Input layer cannot have inputs.')

    def predict(self, xs):
        # yhats, ls, gs
        yhats, _, _ = self(xs)
        return yhats

    def evaluate(self, xs, ys):
        # yhats, ls, gs
        _, ls, _ = self(xs, ys)
        lmean = sum(ls) / len(ls)
        return lmean

    def partial_fit(self, xs, ys, *, alpha=0.03):
        self(xs, ys, alpha)

    def fit(self, xs, ys, *, alpha=0.03, epochs=400):
        for epoch in range(epochs):
            self.partial_fit(xs, ys, alpha=alpha)

class DenseLayer(Layer):

    def __init__(self, outputs, *, name=None, next=None):
        """Initialise a layer with a given amount of outputs and a name."""
        super().__init__(outputs, name=name, next=next)
        self.bias = [0 for _ in range(outputs)]

        # Create weights variable for later use.
        self.weights = None

    def __call__(self, xs, ys=None, alpha=None):
        """
        xs should be a list of lists of values, where each sublist has a number of values equal to self.inputs
        """
        aa = []
        gradients = None
        for x in xs:
            a = []
            for o in range(self.outputs):
                pre_activation = self.bias[o] + sum(self.weights[o][i] * x[i] for i in range(self.inputs))
                a.append(pre_activation)
            aa.append(a)

        yhats, ls, gs = self.next(aa, ys, alpha)

        if alpha:
            gradients = []
            for x, g in zip(xs, gs):
                gradients.append([sum(self.weights[o][i] * g[o] for o in range(self.outputs))
                                  for i in range(self.inputs)])
                for o in range(self.outputs):
                    self.bias[o] -= alpha/len(xs) * g[o]
                    self.weights[o] = [self.weights[o][i] - alpha/len(xs) * g[o] * x[i]
                                       for i in range(self.inputs)]

        return yhats, ls, gradients



    def __repr__(self):
        """Return a string representation of the layer."""
        text = f'DenseLayer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def set_inputs(self, inputs):
        """Set the number of inputs for the layer and initialise the weights.
        Weights are random."""
        self.inputs = inputs
        limit = np.sqrt(6 / (self.inputs + self.outputs))
        if not self.weights:
            self.weights = [[random.uniform(-limit, limit) for _ in range(self.inputs)] for _ in range(self.outputs)]


class ActivationLayer(Layer):
    def __init__(self, outputs, *, name=None, next=None, activation=linear):
        super().__init__(outputs, name=name, next=next)
        self.activation = activation

    def __call__(self, xs, ys=None, alpha=None):
        hh = []   # Uitvoerwaarden voor alle pre activatie waarden berekend in de vorige laag
        grads = None
        for x in xs:
            h = []   # Uitvoerwaarde voor één pre activatie waarde
            for o in range(self.outputs):
                # Bereken voor elk neuron o uit de lijst invoerwaarden x de uitvoerwaarde
                post_activation = self.activation(x[o])
                h.append(post_activation)
            hh.append(h)

        yhats, ls, gs = self.next(hh, ys, alpha)

        if alpha is not None:
            grads = []
            # calculate gradients
            for x, g in zip(xs, gs):
                gg = [derivative(self.activation)(x[i]) * g[i] for i in range(self.inputs)]
                grads.append(gg)


        return yhats, ls, grads
    def __repr__(self):
        text = f'ActivationLayer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text


class LossLayer(Layer):

    def __init__(self, loss=mean_squared_error, name=None):
        super().__init__(name=name,  outputs=0)
        self.loss = loss

    def __repr__(self):
        text = f'LossLayer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __call__(self, xs, ys=None, alpha=None):
        yhats = xs

        # Loss caclulation
        ls = None
        gs = None
        if ys is not None:
            ls = []
            for yhat, y in zip(yhats, ys):
                summed_loss = sum(self.loss(yhat[i], y[i]) for i in range(self.inputs))
                ls.append(summed_loss)

        if alpha is not None:
            gs = []
            for yhat, y in zip(yhats, ys):
                gs.append([derivative(self.loss)(yhat[i], y[i]) for i in range(self.inputs)])
        return yhats, ls, gs
    def __add__(self, next):
        raise NotImplementedError('Cannot add a layer after a loss layer')

# MAIN
def main(args):
    """ Main function """
    xs, ys = data.xorproblem()
    my_network = InputLayer(2) + \
                 DenseLayer(2) + \
                 ActivationLayer(2, activation=sign) + \
                 DenseLayer(1) + \
                 LossLayer()
    my_network[1].bias = [1.0, -1.0]
    my_network[1].weights = [[1.0, 1.0], [1.0, 1.0]]
    my_network[3].bias = [-1.0]
    my_network[3].weights = [[1.0, -1.0]]
    yhats = my_network.predict(xs)
    DataFrame(xs, columns=['x1', 'x2']).assign(y=DataFrame(ys), ŷ=DataFrame(yhats))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))