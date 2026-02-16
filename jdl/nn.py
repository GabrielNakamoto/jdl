from __future__ import annotations
from jdl.tensor import Tensor
import numpy as np
from typing import List

class Layer:
    def __init__(self, inputs, outputs):
        self.shape = (inputs,outputs)
        self.W = Tensor(np.random.normal(0, np.sqrt(2.0/inputs), (inputs,outputs)))
        self.b = Tensor(np.zeros(outputs))
    def forward(self, x: Tensor):
        x = x.reshape((-1, self.shape[0]))
        return x @ self.W + self.b

class Model:
    def __init__(self, inputs, outputs, loss, final, activation=Tensor.relu, layers=[]):
        hidden = len(layers) > 0
        self.layers = []
        self.loss = loss
        self.activation = activation
        self.final = final
        for i, n in enumerate(layers):
            last = inputs if i == 0 else layers[i-1]
            self.layers.append(Layer(last, n))
        self.layers.append(Layer(inputs if not hidden else layers[-1], outputs))
    def params(self):
        params = []
        for l in self.layers: params.extend((l.W,l.b))
        return params
    def forward(self, x, dropout_p=None):
        logits = x
        for l in self.layers[:-1]:
            logits = self.activation(l.forward(logits))
            if dropout_p: logits = logits.dropout(dropout_p)
        return self.final(self.layers[-1].forward(logits))
    def backward(self, pred, y):
        loss = self.loss(pred, y)
        loss.backward()
        return loss

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
        self.step_size = step_size
        self.b1, self.b2 = decay_rates
        self.t = 0
        self.m, self.v = [0.0] * len(params), [0.0] * len(params)
        self.moments = [(0,0) for _ in range(len(params))]
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * p.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * p.grad * p.grad
            m1_hat = self.m[i] / (1 - self.b1 ** self.t)
            v1_hat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.step_size * m1_hat / (np.sqrt(v1_hat) + 1e-8)
