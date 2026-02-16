# Multi-Layer Perceptron Classification Network
import sys
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import mnist_reader
from tensor.tensor import Tensor

class Dataset:
    def __init__(self, training, testing, classes):
        self.X_train, self.y_train = training
        self.X_test, self.y_test = testing
        self.X_train = Tensor(self.X_train.reshape(-1, 28, 28) / 255.0)
        self.X_test = Tensor(self.X_test.reshape(-1, 28, 28) / 255.0)
        self.y_train = Tensor(Dataset.from_1hot(self.y_train, classes))
        self.y_test = Tensor(Dataset.from_1hot(self.y_test, classes))
    @staticmethod
    def from_1hot(y, classes):
        samples, = y.shape
        ny = np.zeros((samples, classes))
        ny[np.arange(samples), y]=1
        return ny
    def sgd_batches(self, batch_size=50):
        N = self.X_train.data.shape[0]
        indices = [i for i in range(N)]
        random.shuffle(indices)
        for i in range(0, N, batch_size):
            j = indices[i:i+batch_size]
            yield Tensor(self.X_train.data.take(j, axis=0)), Tensor(self.y_train.data.take(j, axis=0))


class Layer:
    def __init__(self, inputs, outputs, sigma=0.01):
        self.shape = (inputs,outputs)
        self.W = Tensor(np.random.normal(0, sigma, (inputs,outputs)))
        self.b = Tensor(np.zeros(outputs))
    def forward(self, x: Tensor):
        x = x.reshape((-1, self.shape[0]))
        return x @ self.W + self.b

class Model:
    def __init__(self, inputs, outputs, layers=[], lr=0.1):
        hidden = len(layers) > 0
        self.lr = lr
        self.layers = []
        for i, n in enumerate(layers):
            last = inputs if i == 0 else layers[i-1]
            self.layers.append(Layer(last, n))
        self.layers.append(Layer(inputs if not hidden else layers[-1], outputs))
    def learn(self):
        for l in self.layers:
            l.W.data -= self.lr * l.W.grad
            l.b.data -= self.lr * l.b.grad
    def zero_grads(self):
        for l in self.layers:
            l.W.zero_grad()
            l.b.zero_grad()
    def forward(self, x):
        logits = x
        for l in self.layers[:-1]: logits = l.forward(logits).relu()
        return self.layers[-1].forward(logits).softmax()
    @staticmethod
    def cross_entropy_loss(pred, y):
        # average loss across samples
        return -(y * pred.log()).sum(axis=1).mean()
    def backward(self, pred, y):
        loss = Model.cross_entropy_loss(pred, y)
        loss.backward()
        self.learn()
        return loss


training = mnist_reader.load_mnist('datasets/fashion', kind='train')
testing = mnist_reader.load_mnist('datasets/fashion', kind='t10k')
ds = Dataset(training, testing, 10)

model = Model(784, 10, layers=[256])

# training
for epoch in range(500):
    total_loss = 0.0
    for x, y in ds.sgd_batches(batch_size=200):
        model.zero_grads()
        pred = model.forward(x)
        total_loss += model.backward(pred, y).data
    if epoch % 10 == 0: print(f"Epoch: {epoch}\tloss={total_loss}")

# test generalization
pred = model.forward(ds.X_test)
right = 0
for correct, row in zip(ds.y_test.data, pred.data):
    if row.argmax() == correct.argmax(): right += 1
print(f"{right}/{ds.y_test.data.size} correct classifications")
