# Note: no changes need to be made to this file...

# IMPORTS:

import matplotlib.pyplot as plt
import numpy as np
import warnings, random
from math import pi, cos, sin, sqrt, floor, ceil, atan2, copysign
from cmath import phase


# FUNCTIONS:

def linear(outcome, *, num=100, dim=2, noise=0.0, seed=None):
    """Generate a linear dataset with attributes and outcomes.

    Arguments:
    outcome  -- string indicating 'nominal' or 'numeric' outcomes

    Keyword options:
    num      -- number of instances (default 100)
    dim      -- dimensionality of the attributes (default 2)
    noise    -- the amount of noise to add (default 0.0)
    seed     -- a seed to initialise the random numbers (default random)

    Return values:
    xs       -- values of the attributes
    ys       -- values of the outcomes
    """
    # Check that only one outcome type is specified
    if outcome.lower() != 'nominal' and outcome.lower() != 'numeric':
        raise ValueError('Invalid outcome type specified!')
    # Seed the random number generator
    random.seed(seed)
    # Generate bias and weights
    bias = copysign(1.0, random.gauss(0.0, 1.0))
    weights = [random.gauss(0.0, 1.0) for d in range(dim)]
    norm = sqrt(sum(wi ** 2 for wi in weights))
    for i in range(len(weights)):
        weights[i] /= norm
    # Generate attribute data
    xs = [[random.gauss(0.0, 2.0) for d in range(dim)] for n in range(num)]
    # Generate outcomes
    ys = [bias + sum(wi * xi for wi, xi in zip(weights, x)) for x in xs]
    if outcome.lower() == 'nominal':
        for i in range(len(ys)):
            ys[i] = copysign(1.0, ys[i])
    # Add noise to the attributes
    for n in range(num):
        for d in range(dim):
            xs[n][d] += random.gauss(0.0, noise)
    # Return values
    return xs, ys


def xorproblem(*, num=4, dim=2, noise=0.0, seed=None):
    """Generate a XOR-problem dataset with attributes and outcomes.

    Keyword options:
    num      -- number of instances (default 4)
    dim      -- dimensionality of the attributes (default 2)
    noise    -- the amount of noise to add (default 0.0)
    seed     -- a seed to initialise the random numbers (default random)

    Return values:
    xs       -- values of the attributes
    ys       -- values of the outcomes
    """
    # Seed the random number generator
    random.seed(seed)
    # Generate attribute data
    xs = [[-1.0 if n >> d & 1 == 0 else 1.0 for d in range(dim)] for n in range(num)]
    # Generate outcomes
    ys = [[-1.0] if x.count(1.0) & 1 == 0 else [1.0] for x in xs]
    # Add noise to the attributes
    for n in range(num):
        for d in range(dim):
            xs[n][d] += random.uniform(-noise, noise)
    # Return values
    return xs, ys


def segments(classes, *, num=200, noise=0.0, seed=None):
    """Generate a dataset consisting of circular segments (i.e. "pizza slices").

    Arguments:
    classes  -- number of classes to generate

    Keyword options:
    num      -- number of instances (default 200)
    noise    -- the amount of noise to add (default 0.0)
    seed     -- a seed to initialise the random number generator (default random)

    Return values:
    xs       -- values of the attributes x1 and x2
    ys       -- class labels in one-hot encoding
    """
    # Seed the random number generator
    random.seed(seed)
    # Generate attribute data
    rs = [sqrt(3.0*random.random()) for n in range(num)]
    fs = [random.random() for n in range(num)]
    xs = [[r*cos(2.0*pi*f), r*sin(2.0*pi*f)] for r, f in zip(rs, fs)]
    ys = [[1.0 if floor(f * classes) == c else 0.0 for c in range(classes)] for f in fs]
    # Add noise to the attributes
    for n in range(num):
        for d in range(2):
            xs[n][d] += random.gauss(0.0, noise)
    # Return values
    return xs, ys


def fractal(classes, *, num=200, seed=None):
    """Generate a dataset based on Newton's method applied to 1+(-z)^c=0.

    Arguments:
    classes  -- number of classes to generate

    Keyword options:
    num      -- number of instances (default 200)
    seed     -- a seed to initialise the random number generator (default random)

    Return values:
    xs       -- values of the attributes x1 and x2
    ys       -- class labels in one-hot encoding
    """
    # Seed the random number generator
    random.seed(seed)
    # Generate attribute data
    rs = [sqrt(0.75*random.random()) for n in range(num)]
    fs = [2.0*pi*random.random() for n in range(num)]
    xs = [[r*cos(f), r*sin(f)] for r, f in zip(rs, fs)]
    # Initialize outcomes
    ys = [[0.0 for c in range(classes)] for n in range(num)]
    # Perform Newton's method
    for n in range(num):
        z_old = -complex(xs[n][0], xs[n][1])
        z_new = (z_old*(classes-1)-z_old**(1-classes))/classes
        while abs(z_new-z_old) > 1e-9:
            z_old = z_new
            z_new = (z_old*(classes-1)-z_old**(1-classes))/classes
        c = int(((phase(-z_new)/pi+1.0)*classes-1.0)/2.0)
        ys[n][c] = 1.0
    # Return values
    return xs, ys


def scatter(xs, ys, *, model=None):
    """Plots data according to true and modeled outcomes.

    Arguments:
    xs       -- the values of the attributes
    ys       -- the values of the true outcomes

    Keyword options:
    model    -- the classification/regression model (default None)

    Return values:
    None
    """
    # Wrap ys in list if necessary
    if not isinstance(ys[0], list) and not isinstance(ys[0], np.ndarray):
        ys = [[y] for y in ys]
    # Determine the x-range of the data
    x1s = [xi[0] for xi in xs]
    x2s = [xi[1] for xi in xs]
    xlimit = ceil(1.05 * max(-min(x1s), max(x1s), -min(x2s), max(x2s)))
    xgrid = [i / 64.0 * xlimit for i in range(-64, 65)]
    # Determine the background
    if isinstance(xs, np.ndarray):
        back = model.predict(np.array([[x1, x2] for x2 in xgrid for x1 in xgrid])).reshape((len(xgrid), len(xgrid), -1)).tolist()
    elif hasattr(model, 'predict_proba'):
        back = [[[1.0 - 2.0 * model.predict_proba([[x1, x2]])[0][0]] for x1 in xgrid] for x2 in xgrid]
    elif hasattr(model, 'decision_function'):
        back = [[model.decision_function([[x1, x2]])[0] for x1 in xgrid] for x2 in xgrid]
    elif hasattr(model, 'predict'):
        back = [[model.predict([[x1, x2]])[0] for x1 in xgrid] for x2 in xgrid]
    else:
        back = None
    if back and not isinstance(back[0][0], list):
        back = [[[back[i2][i1]] for i1 in range(129)] for i2 in range(129)]
    # Generate subplots
    axes = len(ys[0])
    fig, axs = plt.subplots(1, axes, figsize=(6.4 * axes, 4.8), squeeze=False)
    for n, ax in enumerate(axs[0]):
        # Determine the y-range of the data
        yns = [yi[n] for yi in ys]
        ylimit = ceil(max(-min(yns), max(yns)))
        # Plot the data
        data = ax.scatter(x1s, x2s, c=yns, edgecolors='w', cmap=plt.cm.RdYlBu, vmin=-ylimit, vmax=ylimit)
        # Background colors denoting the model predictions with dashed line at contour zero
        if back is None:
            ax.set_facecolor('#F8F8F8')
        else:
            backn = [[back_ij[n] for back_ij in back_i] for back_i in back]
            ax.imshow(backn, origin='lower', extent=(-xlimit, xlimit, -xlimit, xlimit), vmin=-ylimit, vmax=ylimit, interpolation='bilinear', cmap=plt.cm.RdYlBu)
            with warnings.catch_warnings():   # Suppress warning that zero-contour may be absent
                warnings.simplefilter('ignore')
                ax.contour(xgrid, xgrid, backn, levels=[0.0], colors='k', linestyles='--', linewidths=1.0)
        # Finish the layout
        ax.set_aspect('equal', 'box')
        ax.axis([-xlimit, xlimit, -xlimit, xlimit])
        ax.grid(True, color='k', linestyle=':', linewidth=0.5)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1.0)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=1.0)
        ax.set_axisbelow(True)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        cbar = plt.colorbar(data, ax=ax).ax
        cbar.axhline(y=0.0, color='k', linestyle='--', linewidth=1.0)
        cbar.set_title(r'$y$' if axes == 1 else r'$y_{}$'.format(n+1))
    plt.show()


def graph(funcs, *args, xlim=(-3.0, 3.0)):
    """Plots the graph of a given function.

    Arguments:
    funcs    -- one or more functions to be plotted
    *args    -- extra arguments that should be passed to the function(s) (optional)

    Keyword options:
    xlim     -- a tuple contain the range of x-values (default (-3.0, 3.0))

    Return values:
    None
    """
    # Wrap the function in a list, if only one is provided
    if not isinstance(funcs, list):
        funcs = [funcs]
    # Plot the figures and keep track of their y-range
    xs = [xlim[0] + i * (xlim[1] - xlim[0]) / 256.0 for i in range(257)]
    ymin = -1.0
    ymax = +1.0
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.subplot(1, 1, 1, facecolor='#F8F8F8')
    for n, func in enumerate(funcs):
        ys = [func(x, *args) for x in xs]
        ymin = min(ymin, floor(min(ys)))
        ymax = max(ymax, ceil(max(ys)))
        plt.plot(xs, ys, color=colors[n % len(colors)], linewidth=3.0, label=func.__name__)
    # Finish the layout
    plt.axis([xlim[0], xlim[1], ymin, ymax])
    plt.legend()
    plt.grid(True, color='k', linestyle=':', linewidth=0.5)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=1.0)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=1.0)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.show()


def curve(series):
    """Plots the curve of a given data series.

    Arguments:
    series   -- a dictionary of data series

    Return values:
    None
    """
    # Plot the curves and keep track of their x-range
    xmax = 1
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for n, label in enumerate(sorted(series.keys())):
        data = series[label]
        xmax = max(xmax, len(data))
        plt.plot([x + 0.5 for x in range(len(data))], data, color=colors[n % len(colors)], linewidth=3.0, label=label)
        plt.axhline(y=min(data), color=colors[n % len(colors)], linewidth=1.0, linestyle='--')
        plt.axhline(y=max(data), color=colors[n % len(colors)], linewidth=1.0, linestyle='--')
    # Finish the layout
    plt.xlim([0, xmax])
    plt.legend()
    plt.grid(True, color='k', linestyle=':', linewidth=0.5)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=1.0)
    plt.xlabel(r'$n$')
    plt.ylabel(r'$y$')
    plt.show()


if __name__ == '__main__':
    raise ImportError('module `data.py` is meant to be imported, not executed')