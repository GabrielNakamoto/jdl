# Multi-Layer Perceptron Classification Network
import sys
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mnist_reader
from jdl import Tensor
from jdl.nn import Linear, ADAM, get_model_params

def sgd_batches(X, y, batch_size=50):
    N = X.data.shape[0]
    indices = [i for i in range(N)]
    random.shuffle(indices)
    for i in range(0, N, batch_size):
        j = indices[i:i+batch_size]
        yield Tensor(X.data.take(j, axis=0)), Tensor(y.data.take(j, axis=0))

def wrap_samples(x, y): return Tensor(x.reshape(-1, 28, 28) / 255.0), Tensor(y)

x_train, y_train = wrap_samples(*mnist_reader.load_mnist('datasets/fashion', kind='train'))
x_test, y_test = wrap_samples(*mnist_reader.load_mnist('datasets/fashion', kind='t10k'))

class Model:
    def __init__(self):
        self.l1 = Linear(784, 256)
        self.l2 = Linear(256, 128)
        self.l3 = Linear(128, 10)
    def __call__(self, x):
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        return self.l3(x)

model = Model()
optimizer = ADAM(model)

# training
for epoch in range(200):
    total_loss = 0.0
    for x, y in sgd_batches(x_train, y_train, batch_size=200):
        optimizer.zero()
        y_hat = model(x)
        loss = y_hat.sparse_categorical_crossentropy(y)
        loss.backward()
        total_loss += loss.data.mean()
        optimizer.step()
    if epoch % 10 == 0: print(f"Epoch: {epoch}\tloss={total_loss}")

# test generalization
pred = model(x_test)
right = 0
for correct, row in zip(y_test.data, pred.data):
    if row.argmax() == correct.argmax(): right += 1
print(f"{right/y_test.data.shape[0]*100:.2f}% correct classifications")
