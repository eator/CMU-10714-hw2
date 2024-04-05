"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.set_bias = bias
        self.weight = Parameter(
                        init.kaiming_uniform(in_features, out_features), 
                        device= device, dtype=dtype
                    )

        if self.set_bias:
            self.bias = Parameter(
                        init.kaiming_uniform(out_features, 1).reshape((1, out_features)), 
                        device= device, dtype=dtype
                    )
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.set_bias:
            #print(X.shape)
            #print(self.weight.shape)
            return X@self.weight + self.bias.broadcast_to((X.shape[0], self.out_features)) 
        else:
            return X@self.weight
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        #dim = len(X.shape) - 1
        #flat = 1
        #for i in range(dim):
        #    flat *= X.shape[i+1]
        #return X.reshape((X.shape[0], flat))
        return X.reshape((X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        #print(self.modules)
        out = x
        for m in self.modules:
            out = m.forward(out)

        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        Z_exp_sum_log_sum = (ops.logsumexp(logits, axes=(1,))).sum()
        y_one_hot = init.one_hot(logits.shape[1], y)
        Z_y_sum = (logits * y_one_hot).sum()

        return (Z_exp_sum_log_sum - Z_y_sum) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        # TODO why can't init by Parameter
        #      because Parameter() increase param nums, but how does this happen?
        #self.running_mean = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=False)) 
        #self.running_var = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=False)) 
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        #print("*****")
        #print(type(self.running_mean))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        #print(f"BatchNorm input_shape: {x.shape}")
        #print(f"dim: {self.dim}")
        m = x.shape[0]
        b = self.momentum

        Ex = x.sum((0,)) / m
        x_Ex = x - Ex.broadcast_to(x.shape)
        Dx = (x_Ex*x_Ex).sum((0,)) / m

        if self.training:
            # TODO why memory like this 
            #self.running_mean = (1-b)*self.running_mean + b*Ex
            #self.running_var = (1-b)*self.running_var + b*Dx

            self.running_mean = (1-b)*self.running_mean + b*Ex.data
            self.running_var = (1-b)*self.running_var + b*Dx.data
            Vx_st = (Dx + self.eps)**(0.5)
            x_norm = x_Ex / Vx_st.broadcast_to(x.shape) 
        else:
            x_Ex_R = x - self.running_mean.broadcast_to(x.shape)
            Vx_st_R = (self.running_var.broadcast_to(x.shape) + self.eps)**(0.5)
            x_norm = x_Ex_R / Vx_st_R
            #x_norm = (x-self.running_mean)/(self.running_var + self.eps)**0.5

        #print("^^^^^")
        #print(self.running_mean.shape)

        return self.weight.broadcast_to(x.shape) * x_norm + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        #print(f"LayerNorm input_shape: {x.shape}")
        #print(f"dim: {self.dim}")
        m = x.shape[0]
        n = x.shape[1]

        Ex = x.sum((1,)).reshape((m, 1)) / n
        x_Ex = x - Ex.broadcast_to(x.shape)
        Dx = (x_Ex*x_Ex).sum((1,)).reshape((m, 1)) / n
        Vx_st = (Dx + self.eps)**(0.5)
        x_norm = x_Ex / Vx_st.broadcast_to(x.shape) 

        return self.weight.broadcast_to(x.shape)*x_norm + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
        #if self.train:
            x_p = x / (1-self.p)
            mask_p = init.randb(*(x.shape), p=1-self.p, dtype="float32")
            return x_p * mask_p
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
