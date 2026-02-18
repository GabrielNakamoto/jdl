# https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mnist_reader
from jdl import Tensor
from jdl.nn import Conv2d, Linear, BatchNorm, ADAM

def sgd_batches(X, y, batch_size=50):
    N = X.data.shape[0]
    indices = [i for i in range(N)]
    random.shuffle(indices)
    for i in range(0, N, batch_size):
        j = indices[i:i+batch_size]
        yield Tensor(X.data.take(j, axis=0)), Tensor(y.data.take(j, axis=0))

def wrap_samples(x, y): return Tensor(x.reshape(-1, 28, 28, 1) / 255.0), Tensor(y)

class Model:
    def __init__(self):
        # Conv block 1
        self.l1 = Conv2d(1, 32, 5)
        self.l2 = Conv2d(32, 32, 5)
        self.l3 = BatchNorm(32)

        # Conv block 2
        self.l4 = Conv2d(32, 64, 3)
        self.l5 = Conv2d(64, 64, 3)
        self.l6 = BatchNorm(64)

        # Fully-connected
        self.l7 = Linear(576, 10)
    def __call__(self, x):
        x = self.l1(x).relu()
        x = self.l2(x)
        x = self.l3(x).relu().max_pool2d((2,2)).dropout(0.25)

        x = self.l4(x).relu()
        x = self.l5(x)
        x = self.l6(x).relu().max_pool2d((2,2)).dropout(0.25).flatten(start=1)
        return self.l7(x)

x_train, y_train = wrap_samples(*mnist_reader.load_mnist('datasets/fashion', kind='train'))
x_test, y_test = wrap_samples(*mnist_reader.load_mnist('datasets/fashion', kind='t10k'))

model = Model()
optimizer = ADAM(model)

batch_size = 200
batches = x_train.shape[0] // batch_size
for epoch in range(20):
    loss = 0.0
    for i, (x, y) in enumerate(sgd_batches(x_train, y_train, batch_size=200)):
        optimizer.zero()
        y_hat = model(x)
        l = y_hat.sparse_categorical_crossentropy(y)
        l.backward()
        loss += l.flatten().mean().data
        optimizer.step()
        print(f"{i}/{batches} batches processed", end='\r', flush=True)
    print(f"Epoch: {epoch}\tloss={loss}")
