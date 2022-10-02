import numpy as np
from itertools import product
from nn import tensor


class Module(object):
    """Base class for all neural network modules.
    """

    def __init__(self) -> None:
        """If a module behaves different between training and testing,
        its init method should inherit from this one."""
        self.training = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Defines calling forward method at every call.
        Should not be overridden by subclasses.
        """
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Defines the forward propagation of the module performed at every call.
        Should be overridden by all subclasses.
        """
        ...

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Defines the backward propagation of the module.
        """
        return dy

    def train(self):
        """Sets the mode of the module to training.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = True
        for attr in vars(self).values():
            if isinstance(attr, Module):
                Module.train()

    def eval(self):
        """Sets the mode of the module to eval.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = False
        for attr in vars(self).values():
            if isinstance(attr, Module):
                Module.eval()


class Linear(Module):

    def __init__(self, in_length: int, out_length: int):
        """Module which applies linear transformation to input.

        Args:
            in_length: L_in from expected input shape (N, L_in).
            out_length: L_out from output shape (N, L_out).
        """

        # w[0] for bias and w[1:] for weight
        self.w = tensor.tensor((in_length + 1, out_length))
        # b = self.w[0]
        # self.w = self.w[1:]

    def forward(self, x):
        """Forward propagation of linear module.

        Args:
            x: input of shape (N, L_in).
        Returns:
            out: output of shape (N, L_out).
        """
        self.x = x
        return np.dot(x, self.w[1:]) + self.w[0]

    def backward(self, dy):
        """Backward propagation of linear module.

        Args:
            dy: output delta of shape (N, L_out).
        Returns:
            dx: input delta of shape (N, L_in).
        """

        return np.dot(dy, self.w[1:].T)


class BatchNorm1d(Module):

    def __init__(self, length: int, momentum: float = 0.9):
        """Module which applies batch normalization to input.

        Args:
            length: L from expected input shape (N, L).
            momentum: default 0.9.
        """
        super(BatchNorm1d, self).__init__()

        # TODO Initialize the attributes
        # of 1d batchnorm module.

        self.momentum = momentum


        # End of todo

    def forward(self, x):
        """Forward propagation of batch norm module.

        Args:
            x: input of shape (N, L).
        Returns:
            out: output of shape (N, L).
        """

        # TODO Implement forward propogation
        # of 1d batchnorm module.

        pass

        # End of todo

    def backward(self, dy):
        """Backward propagation of batch norm module.

        Args:
            dy: output delta of shape (N, L).
        Returns:
            dx: input delta of shape (N, L).
        """

        # TODO Implement backward propogation
        # of 1d batchnorm module.

        pass
        # End of todo


class Conv2d(Module):

    def __init__(self, in_channels: int, channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        """Module which applies 2D convolution to input.

        Args:
            in_channels: C_in from expected input shape (B, C_in, H_in, W_in).
            channels: C_out from output shape (B, C_out, H_out, W_out).
            kernel_size: default 3.
            stride: default 1.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of 2d convolution module.

        pass

        # End of todo

    def forward(self, x):
        """Forward propagation of convolution module.

        Args:
            x: input of shape (B, C_in, H_in, W_in).
        Returns:
            out: output of shape (B, C_out, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of 2d convolution module.

        pass

        # End of todo

    def backward(self, dy):
        """Backward propagation of convolution module.

        Args:
            dy: output delta of shape (B, C_out, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C_in, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of 2d convolution module.

        pass

        # End of todo


class Conv2d_im2col(Conv2d):

    def forward(self, x):
        # TODO Implement forward propogation of
        # 2d convolution module using im2col method.

        pass

        # End of todo


class AvgPool(Module):

    def __init__(self, kernel_size: int = 2,
                 stride: int = 2, padding: int = 0):
        """Module which applies average pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of average pooling module.

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # End of todo

    def forward(self, x):
        """Forward propagation of average pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of average pooling module.

        B, C, H_in, W_in = x.shape
        padding = np.lib.pad(x, pad_width=((1, 1), (1, 1), (1, 1), (1, 1)), mode='constant',
                             constant_values=0)
        H_out = (H_in + 2 * 1 - self.kernel_size) // self.stride + 1
        W_out = (W_in + 2 * 1 - self.kernel_size) // self.stride + 1
        out = np.zeros((B, C, H_out, W_out))
        for b in np.arange(B):
            for c in np.arange(C):
                for i in np.arange(H_out):
                    for j in np.arange(W_out):
                        out[b, c, i, j] = np.mean(padding[b, c,
                                                  self.stride * i:self.stride * i + self.kernel_size,
                                                  self.stride * j:self.stride * j + self.kernel_size])
        return out
        # End of todo

    def backward(self, dy):
        """Backward propagation of average pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of average pooling module.

        return np.repeat(np.repeat(dy, self.stride, axis=2), self.stride, axis=3) / (
                self.kernel_size * self.kernel_size)

        # End of todo


class MaxPool(Module):

    def __init__(self, kernel_size: int = 2,
                 stride: int = 2, padding: int = 0):
        """Module which applies max pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of maximum pooling module.

        self.index = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # End of todo

    def forward(self, x):
        """Forward propagation of max pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of maximum pooling module.

        B, C, H_in, W_in = x.shape
        padding = np.lib.pad(x, pad_width=((1, 1), (1, 1), (1, 1), (1, 1)), mode='constant',
                             constant_values=0)
        H_out = (H_in + 2 * 1 - self.kernel_size) // self.stride + 1
        W_out = (W_in + 2 * 1 - self.kernel_size) // self.stride + 1
        out = np.zeros((B, C, H_out, W_out))
        for b in np.arange(B):
            for c in np.arange(C):
                for i in np.arange(H_out):
                    for j in np.arange(W_out):
                        out[b, c, i, j] = np.max(x[b, c, self.stride * i:self.stride * i + self.kernel_size,
                                                 self.stride * j:self.stride * j + self.kernel_size])
                        index = np.argmax(x[b, c, self.stride * i:self.stride * i + self.kernel_size,
                                          self.stride * j:self.stride * j + self.kernel_size])
                        self.index[
                            b, c, self.stride * i + index // self.kernel_size, self.stride * j + index % self.kernel_size] = 1
        return out

        # End of todo

    def backward(self, dy):
        """Backward propagation of max pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            out: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of maximum pooling module.

        return np.repeat(np.repeat(dy, self.stride, axis=2), self.stride, axis=3) * self.index

        # End of todo


class Dropout(Module):

    def __init__(self, p: float = 0.5):
        # TODO Initialize the attributes
        # of dropout module.
        self.p = p
        self.mask = None

        # End of todo

    def forward(self, x):
        # TODO Implement forward propogation
        # of dropout module.

        if self.training:
            self.mask = np.random.rand(*x.shape) > self.p
            return x * self.mask
        else:
            return x * (1.0 - self.p)

        # End of todo

    def backard(self, dy):
        # TODO Implement backward propogation
        # of dropout module.

        return dy * self.mask

        # End of todo


if __name__ == '__main__':
    import pdb;

    pdb.set_trace()
