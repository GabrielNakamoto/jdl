# https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import mnist_reader
from jdl import Tensor
from jdl.nn import Conv2d, Linear, BatchNorm, ADAM

def sgd_batches(X, y, batch_size=50, drop_last=False, augment=False):
    N = X.data.shape[0]
    indices = list(range(N))
    random.shuffle(indices)
    for i in range(0, N, batch_size):
        j = indices[i:i+batch_size]
        if drop_last and len(j) < batch_size: continue
        x_batch = X.data.take(j, axis=0)
        if augment:
            # Random horizontal flip
            flip_mask = np.random.random(len(j)) > 0.5
            x_batch[flip_mask] = x_batch[flip_mask, :, ::-1, :]
        yield Tensor(x_batch), Tensor(y.data.take(j, axis=0))

def wrap_samples(x, y): return Tensor(x.reshape(-1, 28, 28, 1) / 255.0), Tensor(y)

class Model:
    def __init__(self):
        self.l1 = Conv2d(1, 32, 3)
        self.l2 = BatchNorm(32)

        self.l3 = Conv2d(32, 64, 3)
        self.l4 = BatchNorm(64)

        self.l5 = Linear(64 * 5 * 5, 128)
        self.l6 = Linear(128, 10)
        self.training = True
    def train(self): self.training = True
    def eval(self): self.training = False
    def __call__(self, x):
        p = 0.25 if self.training else 0.0  # dropout probability
        x = self.l1(x)
        x = self.l2(x).relu().max_pool2d((2,2)).dropout(p)

        x = self.l3(x)
        x = self.l4(x).relu().max_pool2d((2,2)).flatten(start=1).dropout(p)

        x = self.l5(x).relu().dropout(0.5 if self.training else 0.0)
        return self.l6(x)

x_train, y_train = wrap_samples(*mnist_reader.load_mnist('datasets/fashion', kind='train'))
x_test, y_test = wrap_samples(*mnist_reader.load_mnist('datasets/fashion', kind='t10k'))

model = Model()
optimizer = ADAM(model, step_size=0.001)
weight_decay = 1e-4  # L2 regularization

batch_size = 64
batches = x_train.shape[0] // batch_size
for epoch in range(50):
    model.train()
    loss = 0.0
    for i, (x, y) in enumerate(sgd_batches(x_train, y_train, batch_size=batch_size, drop_last=True, augment=True)):
        optimizer.zero()
        y_hat = model(x)
        l = y_hat.sparse_categorical_crossentropy(y).backward()
        # Apply weight decay (L2 regularization)
        for p in optimizer.params:
            p.grad += weight_decay * p.data
        loss += l.flatten().mean().data[0]
        optimizer.step()
        print(f"{i+1}/{batches} batches processed", end='\r', flush=True)
    model.eval()
    y_pred = model(x_test)
    acc = (y_pred.data.argmax(axis=1) == y_test.data).mean()
    print(f"Epoch: {epoch}\tloss={loss/batches:.4f}\ttest_acc={acc*100:.2f}%")
