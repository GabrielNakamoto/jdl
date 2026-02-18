import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mnist_reader
from jdl import Tensor
from jdl.nn import Linear, Conv2d, AvgPool, SGD

def sgd_batches(X, y, batch_size=50):
    N = X.data.shape[0]
    indices = [i for i in range(N)]
    random.shuffle(indices)
    for i in range(0, N, batch_size):
        j = indices[i:i+batch_size]
        yield Tensor(X.data.take(j, axis=0)), Tensor(y.data.take(j, axis=0))

class Model:
    def __init__(self):
        self.l1 = Conv2d(1, 6, 5, padding=2)
        self.l2 = AvgPool(2, 2)
        self.l3 = Conv2d(6, 16, 5)
        self.l4 = AvgPool(2, 2)

        self.l5 = Linear(400, 120)
        self.l6 = Linear(120, 84)
        self.l7 = Linear(84, 10)
    def __call__(self, x):
        x = self.l1(x).sigmoid()
        x = self.l2(x)
        x = self.l3(x).sigmoid()
        x = self.l4(x).flatten()
        x = self.l5(x).sigmoid()
        x = self.l6(x).sigmoid()
        return self.l7(x).softmax()

def cross_entropy_loss(y_hat, y): return -(y * y_hat.log()).sum(axis=1).mean()
def wrap_samples(x, y): return Tensor(x.reshape(-1, 28, 28, 1) / 255.0), Tensor(y).one_hot(10)

x_train, y_train = wrap_samples(*mnist_reader.load_mnist('datasets/fashion', kind='train'))
x_test, y_test = wrap_samples(*mnist_reader.load_mnist('datasets/fashion', kind='t10k'))

model = Model()
optimizer = SGD(model)

for epoch in range(20):
    total_loss = 0.0
    for x, y in sgd_batches(x_train, y_train, batch_size=200):
        optimizer.zero()
        y_hat = model(x)
        l = cross_entropy_loss(y_hat, y)
        l.backward()
        total_loss += l.flatten().mean().data
        optimizer.step()
    print(f"Epoch: {epoch}\tloss={total_loss}")
