import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from os import wait
import random
from tensor.tensor import Tensor
import numpy as np
import mnist_reader

class Dataset:
    def __init__(self, training, testing):
        self.X_train, self.y_train = training
        self.X_test, self.y_test = testing
        self.X_train = Tensor(self.X_train.reshape(-1, 28, 28) / 255.0)
        self.X_test = Tensor(self.X_test.reshape(-1, 28, 28) / 255.0)
        self.y_train = Tensor(self.y_train)
        self.y_test = Tensor(self.y_test)
    def sgd_batches(self, batch_size=50):
        N = self.X_train.data.shape[0]
        indices = [i for i in range(N)]
        random.shuffle(indices)
        for i in range(0, N, batch_size):
            j = indices[i:i+batch_size]
            yield Tensor(self.X_train.data.take(j, axis=0)), Tensor(self.y_train.data.take(j, axis=0))

class Model:
    def __init__(self, features, classes, batch_size=50, lr=0.03, sigma=0.01):
        self.lr = lr
        self.features = features
        self.weights = Tensor(np.random.normal(0, sigma, size=(features, classes)))
        self.bias = Tensor(np.zeros(classes))
        self.one_hot_buffer = np.zeros((batch_size, classes))
        self.hot_indices = np.arange(batch_size)
        self.epsilon = 1e-8
    def forward(self, X):
        X = X.reshape((-1, self.features))
        O = X @ self.weights + self.bias
        return O.softmax()
    def zero_grad(self):
        self.weights.zero_grad()
        self.bias.zero_grad()
    def optim(self):
        self.weights.data -= self.lr * self.weights.grad
        self.bias.data -= self.lr * self.bias.grad
    def cross_entropy_loss(self, pred, y):
        self.one_hot_buffer.fill(0)
        self.one_hot_buffer[self.hot_indices, y.data.astype(int)] = 1.0
        one_hot = Tensor(self.one_hot_buffer)
        log_pred = (pred + Tensor(np.array(self.epsilon))).log()
        return -(log_pred * one_hot).sum(axis=1).mean()

training = mnist_reader.load_mnist('datasets/fashion', kind='train')
testing = mnist_reader.load_mnist('datasets/fashion', kind='t10k')

ds = Dataset(training, testing)
model = Model(784, 10, lr=0.1, batch_size=200)

for epoch in range(200):
    total_loss = 0.0
    for x, y in ds.sgd_batches(batch_size=200):
        model.zero_grad()
        pred = model.forward(x)
        loss = model.cross_entropy_loss(pred, y)
        loss.backward()
        model.optim()
        total_loss += loss.data.mean()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}\tloss={total_loss:.2f}")
