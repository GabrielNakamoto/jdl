from __future__ import annotations
from jdl.tensor import Tensor
import numpy as np
from typing import List


def get_model_params(model):
    params = []
    for _, value in vars(model).items():
        if isinstance(value, Linear): params.extend((value.W, value.b))
    return params

class Linear:
    def __init__(self, inputs, outputs):
        self.shape = (inputs,outputs)
        self.W = Tensor(np.random.normal(0, np.sqrt(2.0/inputs), (inputs,outputs)))
        self.b = Tensor(np.zeros(outputs))
    def __call__(self, x: Tensor):
        x = x.reshape((-1, self.shape[0]))
        return x @ self.W + self.b

class Optimizer:
    def __init__(self, params: List[Tensor]):
        self.params = params
    def zero(self):
        for p in self.params: p.zero_grad()


class SGD(Optimizer):
    def __init__(self, params: List[Tensor], lr=0.01):
        super().__init__(params)
        self.lr = lr
    def step(self):
        for p in self.params: p.data -= self.lr * p.grad

# https://arxiv.org/pdf/1412.6980
class ADAM(Optimizer):
    def __init__(self, params: List[Tensor], step_size=0.001, decay_rates=(0.9,0.999)):
        super().__init__(params)
        self.step_size, self.t = step_size, 0
        self.b1, self.b2 = decay_rates
        self.m, self.v = [0.0] * len(params), [0.0] * len(params)
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * p.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * p.grad * p.grad
            m1_hat = self.m[i] / (1 - self.b1 ** self.t)
            v1_hat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.step_size * m1_hat / (np.sqrt(v1_hat) + 1e-8)
