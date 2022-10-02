import random
import argparse
import functools
import inspect
import sys
import time

import numpy as np
import torch
import nn.functional as F

import nn
import module_test

if __name__ == '__main__':
    # a = nn.Linear
    # b = module_test.LinearTest()
    # output = a(b.forward_test)
    # dx = a.backward(output)
    x = np.array([[[[1, 2, 3]]], [[[4, 5, 6]]]])
    B, C, H_in, W_in = x.shape
    padding = np.lib.pad(x, pad_width=((1, 1), (1, 1), (1, 1), (1, 1)), mode='constant',
                         constant_values=0)
    H_out = (H_in + 2 * 1 - 2) // 2 + 1
    W_out = (W_in + 2 * 1 - 2) // 2 + 1

    out = np.zeros((B, C, H_out, W_out))
    print(out)
    # for b in np.arange(B):
    #     for c in np.arange(C):
    #         for i in np.arange(H_out):
    #             for j in np.arange(W_out):
    #                 out[b, c, i, j] = np.mean(padding[b, c,
    #                                           2 * i:2 * i + 2,
    #                                           2 * j:2 * j + 2])
    # print(padding)
