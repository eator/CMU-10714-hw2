from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z_max = array_api.max(Z, self.axes)

        if self.axes == None or len(self.axes) == len(Z.shape): 
            Z_max_expand = array_api.broadcast_to(Z_max, Z.shape)
        else:
            new_list = list(Z.shape)
            for i in range(len(self.axes)):
                new_list[self.axes[i]] = 1

            new_shape = tuple(new_list)
            Z_max_expand = array_api.broadcast_to(array_api.reshape(Z_max, new_shape), Z.shape)

        return array_api.log(array_api.sum(array_api.exp(Z - Z_max_expand), self.axes)) + Z_max
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0].cached_data
        Z_max = array_api.max(Z, self.axes)

        if self.axes == None or len(self.axes) == len(Z.shape): 
            Z_max_expand = array_api.broadcast_to(Z_max, Z.shape)
        else:
            new_list = list(Z.shape)
            for i in range(len(self.axes)):
                new_list[self.axes[i]] = 1

            new_shape = tuple(new_list)
            Z_max_expand = array_api.broadcast_to(array_api.reshape(Z_max, new_shape), Z.shape)

        s = array_api.sum(array_api.exp(Z - Z_max_expand), self.axes)
        if self.axes == None or len(self.axes) == len(Z.shape): 
            s_expand = array_api.broadcast_to(s, Z.shape)
            out_grad_expand = out_grad.broadcast_to(Z.shape)
        else:
            new_list = list(Z.shape)
            for i in range(len(self.axes)):
                new_list[self.axes[i]] = 1

            new_shape = tuple(new_list)
            s_expand = array_api.broadcast_to(array_api.reshape(s, new_shape), Z.shape)
            out_grad_expand = out_grad.reshape(new_shape).broadcast_to(Z.shape)

        #print("\n")
        #print(Z_max_expand.shape)
        #print(s_expand.shape)
        #print(out_grad_expand.shape)
        res =  out_grad_expand * (exp(Tensor(Z - Z_max_expand)) / Tensor(s_expand)) 
        #print(res.shape)
        return res
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

