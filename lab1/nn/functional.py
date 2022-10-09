import math

import numpy as np
from numba.cuda import target

from mnist import n_classes, n_epochs
from .modules import Module


class Sigmoid(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # TODO Implement forward propogation
        # of sigmoid function.

        self.y = 1.0 / (1 + np.exp(-x))
        return self.y
        # End of todo

    def backward(self, dy):
        # TODO Implement backward propogation
        # of sigmoid function.
        return (1 - self.y) * self.y * dy

        # End of todo


class Tanh(Module):

    def forward(self, x):
        # TODO Implement forward propogation
        # of tanh function.

        self.x = x
        return np.tanh(x)
        # End of todo

    def backward(self, dy):
        # TODO Implement backward propogation
        # of tanh function.

        return (1 - np.square(self.x)) * dy

        # End of todo


class ReLU(Module):

    def forward(self, x):
        # TODO Implement forward propogation
        # of ReLU function.

        self.x = x
        return np.maximum(0, x)

        # End of todo

    def backward(self, dy):
        # TODO Implement backward propogation
        # of ReLU function.

        dy[self.x < 0] = 0
        return dy

        # End of todo


class Softmax(Module):

    def forward(self, x):
        # TODO Implement forward propogation
        # of Softmax function.
        x -= np.max(x)
        return np.exp(x) / np.sum(np.exp(x))
        # End of todo

    def backward(self, dy):
        # Omitted.
        ...


class Loss(object):
    """
    Usage:
        >>> criterion = Loss(n_classes)
        >>> ...
        >>> for epoch in n_epochs:
        ...     ...
        ...     probs = model(x)
        ...     loss = criterion(probs, target)
        ...     model.backward(loss.backward())
        ...     ...
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, probs, targets):
        self.probs = probs
        self.targets = targets
        ...
        return self

    def backward(self):
        ...


class SoftmaxLoss(Loss):

    def __call__(self, probs, targets):
        # TODO Calculate softmax loss.

        super(SoftmaxLoss, self).__call__(probs, targets)
        exps = np.exp(probs)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.value = np.sum(-np.eye(self.n_classes)[targets] * np.log(probs))
        return self

        # End of todo

    def backward(self):
        # TODO Implement backward propogation
        # of softmax loss function.

        return self.probs - np.eye(self.n_classes)[self.targets]

        # End of todo


class CrossEntropyLoss(SoftmaxLoss):

    def __call__(self, probs, targets):
        # TODO Calculate cross-entropy loss.

        super(SoftmaxLoss, self).__call__(probs, targets)
        return self

        # End of todo

    def backward(self):
        # TODO Implement backward propogation
        # of cross-entropy loss function.

        ...

        # End of todo

