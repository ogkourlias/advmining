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
    """Linear activation function."""
    return a


def sign(a):
    """Sign activation function."""
    return np.sign(a)


def tanh(a):
    """Hyperbolic tangent activation function."""
    return np.tanh(a)


# Loss functions
def mean_squared_error(yhat, y):
    """Mean squared error loss function."""
    return np.square(yhat - y)


def mean_absolute_error(yhat, y):
    """Mean absolute error loss function."""
    return yhat - y


def derivative(function, delta=0.001):
    """Return the derivative of a function.
    Using a wrapper function to preserve the function's name and qualname."""

    def wrapper_derivative(x, *args):
        """Return the derivative of a function."""
        return (function(x + delta, *args) - function(x - delta, *args)) / (2 * delta)
        wrapper_derivative.__name__ = function.__name__ + '’'
        wrapper_derivative.__qualname__ = function.__qualname__ + '’'

    return wrapper_derivative


# CLASSES
class Neuron:
    """A neuron with a given amount of inputs and a given activation function."""

    def __init__(self, dim, activation=linear, loss=mean_squared_error, bias=0):
        """Initialise a neuron with a given amount of inputs and a given activation function."""
        self.dim = dim
        self.activation = activation
        self.loss = loss
        self.bias = bias
        self.weights = [0 for _ in range(dim)]
        self.loss_derivs = []

    def __repr__(self):
        """Return a string representation of the neuron."""
        text = f'Neuron(dim={self.dim}, activation={self.activation.__name__}, loss={self.loss.__name__})'
        return text

    def predict(self, xs):
        """Return the predicted values for a given set of inputs."""
        predicted = []
        for xvals in xs:  # For each input in instance.
            pre = self.bias + sum(self.weights[i] * xvals[i] for i in range(self.dim))  # Calculate the pre-activation.
            post = self.activation(pre)  # Calculate the post-activation.
            predicted.append(post)  # Add the post-activation to the list of predictions.
        return predicted

    def partial_fit(self, xs, ys, *, alpha=0.03):
        """Perform a partial fit on the neuron."""
        for xvals, y in zip(xs, ys):  # For each input in instance.
            pre = self.bias + sum(self.weights[i] * xvals[i] for i in range(self.dim))  # Calculate the pre-activation.
            yhat = self.activation(pre)  # Calculate the post-activation.
            self.loss_derivs.append(derivative(self.loss)(yhat, y))  # Calculate the loss derivative.
            self.bias -= alpha * derivative(self.loss)(yhat, y) * derivative(self.activation)(pre)  # Update the bias.
            for i in range(self.dim):  # Update the weights.
                self.weights[i] -= alpha * derivative(self.loss)(yhat, y) * derivative(self.activation)(pre) * xvals[i]

    def fit(self, xs, ys, *, alpha=0.03, epochs=4000):
        """Perform a fit on the neuron."""
        for epoch in range(epochs):  # For each epoch.
            self.partial_fit(xs, ys)  # Perform a partial fit.
            if np.average(self.loss_derivs) <= 0.03:  # If the average loss derivative is less than 0.03.
                self.loss_derivs = []  # Reset the loss derivatives.
                break


class Layer:
    """A layer of neurons."""
    classcounter = Counter()  # Counter for the number of layers of each type.

    def __init__(self, outputs, *, name=None, next=None):
        """Initialise a layer with a given amount of outputs and a name."""
        Layer.classcounter[type(self)] += 1  # Increment the counter for the layer type.
        if name is None:  # If no name is given.
            name = f'{type(self).__name__}_{Layer.classcounter[type(self)]}'  # Create a name for the layer.
        self.inputs = 0  # The number of inputs is initially 0.
        self.outputs = outputs  # The number of outputs is the given number of outputs.
        self.name = name  # The name of the layer is the given name.
        self.next = next  # The next layer is the given next layer.

    def __add__(self, next):
        """Add a layer to the current layer. (Dunder method)"""
        result = deepcopy(self)  # Create a copy of the current layer.
        result.add(deepcopy(next))  # Add the next layer to the copy.
        return result  # Return the copy.

    def __getitem__(self, index):
        """Return the layer at the given index. (Dunder method)"""
        if index == 0 or index == self.name:  # If the index is 0 or the name of the layer.
            return self  # Return the layer.
        if isinstance(index, int):  # If the index is an integer.
            if self.next is None:  # If there is no next layer.
                raise IndexError('Layer index out of range')  # Raise an error.
            return self.next[index - 1]  # Return the layer at the given index.
        if isinstance(index, str):  # If the index is a string.
            if self.next is None:  # If there is no next layer.
                raise KeyError(index)  # Raise an error.
            return self.next[index]  # Return the layer with the given name.
        raise TypeError(f'Layer indices must be integers or strings, not {type(index).__name__}')  # Raise an error.

    def __call__(self, xs):
        """Return the output of the layer. (Dunder method)"""
        raise NotImplementedError('Abstract __call__ method')

    def __repr__(self):
        """Return a string representation of the layer."""
        text = f'Layer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:  # If there is a next layer.
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
        """Set the number of inputs of the layer."""
        self.inputs = inputs


class InputLayer(Layer):
    """Input layer of a neural network."""

    def __call__(self, xs, ys=None, alpha=None):
        """Return the output of the layer. (Dunder method)"""
        return self.next(xs, ys, alpha)

    def __repr__(self):
        """Return a string representation of the layer."""
        text = f'InputLayer(outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            # If there is a next layer.
            text += ' + ' + repr(self.next)
        return text

    def set_inputs(self, inputs):
        """Set the number of inputs of the layer."""
        raise AttributeError('Input layer cannot have inputs.')

    def predict(self, xs):
        """Predict for yhats."""
        # yhats, ls, gs
        yhats, _, _ = self(xs)
        return yhats

    def evaluate(self, xs, ys):
        """Evaluate the model using loss values."""
        # yhats, ls, gs
        _, ls, _ = self(xs, ys)
        lmean = sum(ls) / len(ls)
        return lmean

    def partial_fit(self, xs, ys, *, alpha=0.03):
        """Perform a partial fit on the model."""
        self(xs, ys, alpha)

    def fit(self, xs, ys, *, alpha=0.03, epochs=400):
        """Perform a fit on the model."""
        for epoch in range(epochs):
            self.partial_fit(xs, ys, alpha=alpha)


class DenseLayer(Layer):
    """A hidden dense layer of neurons."""

    def __init__(self, outputs, *, name=None, next=None):
        """Initialise a layer with a given amount of outputs and a name."""
        super().__init__(outputs, name=name, next=next)
        self.bias = [0 for _ in range(outputs)]

        # Create weights variable for later use.
        self.weights = None

    def __call__(self, xs, ys=None, alpha=None):
        """Call the layer. (Dunder method)"""
        aa = []  # Create an empty list for outputs.
        gradients = None  # Create a variable for gradients.
        for x in xs:  # For each input in each instance.
            a = []  # Create an empty list for outputs.
            for o in range(self.outputs):  # For each output.
                # Calculate the pre-activation.
                pre_activation = self.bias[o] + \
                                 sum(self.weights[o][i] * x[i] for i in range(self.inputs))
                a.append(pre_activation)  # Append the pre-activation to the list of outputs.
            aa.append(a)  # Append the list of instance outputs to the list of outputs.

        yhats, ls, gs = self.next(aa, ys, alpha)  # Call the next layer.

        if alpha:  # If alpha is given the training is happening.
            gradients = []  # Create an empty list for gradients.
            for x, g in zip(xs, gs):  # For each instance and its gradient.
                gradients.append([sum(self.weights[o][i] * g[o] for o in range(self.outputs))
                                  for i in range(self.inputs)])  # Calculate the gradient.
                for o in range(self.outputs):  # For each output.
                    self.bias[o] -= alpha / len(xs) * g[o]  # Update the bias.
                    self.weights[o] = [self.weights[o][i] - alpha / len(xs) * g[o] * x[i]
                                       for i in range(self.inputs)]  # Update the weights.

        return yhats, ls, gradients

    def __repr__(self):
        """Return a string representation of the layer."""
        text = f'DenseLayer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'  # Create a string.
        if self.next is not None:  # If there is a next layer.
            text += ' + ' + repr(self.next)  # Add the next layer to the string.
        return text

    def set_inputs(self, inputs):
        """Set the number of inputs for the layer and initialise the weights.
        Weights are random."""
        self.inputs = inputs  # Set the number of inputs.
        limit = np.sqrt(6 / (self.inputs + self.outputs))  # Calculate the limit for the random weights.
        if not self.weights:
            self.weights = [[random.uniform(-limit, limit) for _ in range(self.inputs)] for _ in range(self.outputs)]


class ActivationLayer(Layer):
    """A layer that applies an activation function to its inputs."""

    def __init__(self, outputs, *, name=None, next=None, activation=linear):  # Add activation function as parameter.
        super().__init__(outputs, name=name, next=next)  # Initialise the layer.
        self.activation = activation  # Set the activation function.

    def __call__(self, xs, ys=None, alpha=None):
        """Call the layer. (Dunder method)"""
        hh = []  # Uitvoerwaarden voor alle pre activatie waarden berekend in de vorige laag
        grads = None
        for x in xs:
            h = []  # Uitvoerwaarde voor één pre activatie waarde
            for o in range(self.outputs):
                # Bereken voor elk neuron o uit de lijst invoerwaarden x de uitvoerwaarde
                post_activation = self.activation(x[o])
                h.append(post_activation)
            hh.append(h)  # Voeg de uitvoerwaarde toe aan de lijst met uitvoerwaarden

        yhats, ls, gs = self.next(hh, ys, alpha)  # Roep de volgende laag aan

        if alpha is not None:  # Als alpha is meegegeven, dan is er training
            grads = []  # lijst met de gradienten van de pre activatie waarden
            # calculate gradients
            for x, g in zip(xs, gs):  # Voor elke invoer en de bijbehorende gradient
                gg = [derivative(self.activation)(x[i]) * g[i] for i in range(self.inputs)]
                grads.append(gg)  # Voeg de gradient toe aan de lijst met gradienten

        return yhats, ls, grads

    def __repr__(self):
        """Return a string representation of the layer."""
        text = f'ActivationLayer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text


class LossLayer(Layer):
    """A layer that calculates the loss."""

    def __init__(self, loss=mean_squared_error, name=None):
        super().__init__(name=name, outputs=0)
        self.loss = loss

    def __repr__(self):
        """Return a string representation of the layer."""
        text = f'LossLayer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __call__(self, xs, ys=None, alpha=None):
        """Call the layer. (Dunder method)"""
        yhats = xs  # De uitvoerwaarden van de vorige laag zijn de invoerwaarden van de loss laag
        ls = None  # De loss is nog niet berekend
        gs = None  # De gradient is nog niet berekend
        if ys is not None:  # Als er een lijst met gewenste uitvoerwaarden is meegegeven
            ls = []  # Maak een lege lijst voor de loss
            for yhat, y in zip(yhats, ys):  # Voor elke uitvoerwaarde en de bijbehorende gewenste uitvoerwaarde
                summed_loss = sum(self.loss(yhat[i], y[i]) for i in range(self.inputs))
                ls.append(summed_loss)  # Voeg de loss toe aan de lijst met lossen

        if alpha is not None:  # Als er een alpha is meegegeven, dan is er training
            gs = []  # Maak een lege lijst voor de gradienten
            for yhat, y in zip(yhats, ys):  # Voor elke uitvoerwaarde en de bijbehorende gewenste uitvoerwaarde
                # Voeg de gradient toe aan de lijst met gradienten
                gs.append([derivative(self.loss)(yhat[i], y[i]) for i in range(self.inputs)])
        return yhats, ls, gs

    def __add__(self, next):
        """Add a layer to the loss layer."""
        raise NotImplementedError('Cannot add a layer after a loss layer')



def main(args):
    """ Main function """

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
