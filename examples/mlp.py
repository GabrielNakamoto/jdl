# Multi-Layer Perceptron Classification Network
import sys
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import mnist_reader
from jdl import Tensor
from jdl.nn import Layer, Model

def sgd_batches(X, y, batch_size=50):
    N = X.data.shape[0]
    indices = [i for i in range(N)]
    random.shuffle(indices)
    for i in range(0, N, batch_size):
        j = indices[i:i+batch_size]
        yield Tensor(X.data.take(j, axis=0)), Tensor(y.data.take(j, axis=0))

def from_1hot(y, classes):
    samples, = y.shape
    ny = np.zeros((samples, classes))
    ny[np.arange(samples), y]=1
    return ny

def cross_entropy_loss(pred, y):
    return -(y * pred.log()).sum(axis=1).mean()

def wrap_samples(x, y):
    x = Tensor(x.reshape(-1, 28, 28) / 255.0)
    y = Tensor(from_1hot(y, 10))
    return (x, y)

x_train, y_train = wrap_samples(*mnist_reader.load_mnist('datasets/fashion', kind='train'))
x_test, y_test = wrap_samples(*mnist_reader.load_mnist('datasets/fashion', kind='t10k'))

model = Model(784, 10, cross_entropy_loss, Tensor.softmax, layers=[256])

# training
for epoch in range(500):
    total_loss = 0.0
    for x, y in sgd_batches(x_train, y_train ,batch_size=200):
        model.zero_grads()
        pred = model.forward(x)
        total_loss += model.backward(pred, y).data
    if epoch % 10 == 0: print(f"Epoch: {epoch}\tloss={total_loss}")

# test generalization
pred = model.forward(x_test)
right = 0
for correct, row in zip(y_test.data, pred.data):
    if row.argmax() == correct.argmax(): right += 1
print(f"{right/y_test.data.shape[0]*100:.2f}% correct classifications")
