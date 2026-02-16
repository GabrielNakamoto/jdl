from jdl.tensor import Tensor
import numpy as np

class Layer:
    def __init__(self, inputs, outputs, sigma=0.01):
        self.shape = (inputs,outputs)
        self.W = Tensor(np.random.normal(0, sigma, (inputs,outputs)))
        self.b = Tensor(np.zeros(outputs))
    def forward(self, x: Tensor):
        x = x.reshape((-1, self.shape[0]))
        return x @ self.W + self.b

class Model:
    def __init__(self, inputs, outputs, loss, final, activation=Tensor.relu, layers=[], lr=0.1):
        hidden = len(layers) > 0
        self.lr = lr
        self.layers = []
        self.loss = loss
        self.activation = activation
        self.final = final
        for i, n in enumerate(layers):
            last = inputs if i == 0 else layers[i-1]
            self.layers.append(Layer(last, n))
        self.layers.append(Layer(inputs if not hidden else layers[-1], outputs))
    def parameters(self):
        params = []
        for l in self.layers: params.extend((l.W,l.b))
        return params
    def learn(self):
        for p in self.parameters(): p.data -= self.lr * p.grad
    def zero_grads(self):
        for p in self.parameters(): p.zero_grad()
    def forward(self, x):
        logits = x
        for l in self.layers[:-1]: logits = self.activation(l.forward(logits))
        return self.final(self.layers[-1].forward(logits))
    def backward(self, pred, y):
        loss = self.loss(pred, y)
        loss.backward()
        self.learn()
        return loss
