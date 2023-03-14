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
                y_roof += self.bias + x[dim]*self.weights[dim]
                self.weights[dim] -= alpha * (y_roof-y) * x[dim]
            print(self.bias)
            self.bias -= alpha * (y_roof-y)
            # Update hier het perceptron met één instance {x, y}

    def fit(self, xs, ys, *, epochs=1000, alpha=0.01):
        if epochs != 0:
            for epoch in range(epochs):
                old_bias = self.bias
                old_wght = self.weights
                self.partial_fit(xs, ys)
                wgth_diff = []
                for i, j in zip(old_wght, self.weights):
                    wgth_diff.append(min(i,j) / max(i, j))
                if 0.025 * old_bias <= abs(old_bias-self.bias) and max(wgth_diff) <= 1 - alpha/2:
                    break

def linear(a):
    return a

def sign(a):
    return a

def mean_squared_error(yhat, y):
    return numpy.square(yhat-y)

def mean_absolute_error(yhat, y):
    return yhat - y


def derivative(function, delta=...):
    def wrapper_derivative(x):
        return function(x)



        wrapper_derivative.__name__ = function.__name__ + '’'
        wrapper_derivative.__qualname__ = function.__qualname__ + '’'

    return wrapper_derivative