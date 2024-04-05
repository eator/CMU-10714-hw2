"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        m = self.momentum
        w = self.weight_decay
        lr = self.lr
        for i, param in enumerate(self.params):
            if i not in self.u:
                self.u[i] = ndl.init.zeros_like(param, requires_grad=True)

            #print("*****")
            #print(param.grad.dtype)
            #print(param.dtype)
            l2_grad = ndl.Tensor(param.grad, dtype="float32") + w*param
            self.u[i] = m*self.u[i] + (1-m)*l2_grad
            param.data = param - lr*self.u[i] 
            # TODO
            # param = param - lr*self.u[i] # why this can't update param 
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        w = self.weight_decay
        b1 = self.beta1
        b2 = self.beta2
        lr = self.lr
        eps = self.eps

        self.t +=1

        for i, param in enumerate(self.params):
            if i not in self.m:
                self.m[i] = ndl.init.zeros_like(param, requires_grad=True)
                self.v[i] = ndl.init.zeros_like(param, requires_grad=True)

            l2_grad = ndl.Tensor(param.grad, dtype="float32") + w*param
            self.m[i] = b1*self.m[i] + (1-b1)*l2_grad
            self.v[i] = b2*self.v[i] + (1-b2)*l2_grad*l2_grad
            m_bc = self.m[i] / (1-b1**self.t) 
            v_bc = self.v[i] / (1-b2**self.t) 
            param.data = param - lr*m_bc/(v_bc**0.5 + eps) 
        ### END YOUR SOLUTION
