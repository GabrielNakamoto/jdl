from sklearn import preprocessing
from tensor import Tensor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class Dataset:
    def __init__(self, samples, features, X, y):
        self.n = samples
        self.features = features
        self.X = Tensor(X)
        self.y = Tensor(y)
    def stochastic_minibatcher(self, batch_size):
        indices = list(range(0, self.n))
        random.shuffle(indices)
        for i in range(0, self.n, batch_size):
            j = indices[i:i+batch_size]
            yield Tensor(self.X.data.take(j, axis=0)), Tensor(self.y.data.take(j, axis=0))

def syntheticRegressionData(w, b, feature_shape, samples, noise=0.01):
    shape = (samples,) + feature_shape
    X = np.random.randn(*shape)
    z = np.random.randn(samples) * noise
    y = X @ w.T + b + z
    return Dataset(samples, feature_shape[0], X, y.reshape(-1, 1))

def carPriceDataset():
    cat_cols = ['CarName', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']
    num_cols = ['symboling', 'curbweight', 'enginesize', 'peakrpm', 'citympg', 'highwaympg']
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ], sparse_threshold=0)
    df = pd.read_csv("CarPrice_Assignment.csv")

    y_raw = df['price'].to_numpy().reshape(-1, 1)
    df = df.drop('price', axis=1)
    x = preprocessor.fit_transform(df)

    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y_raw)

    samples, features = x.shape
    return Dataset(samples, features, x, y)


class Model:
    def __init__(self, x_dims, lr, sigma=0.01):
        self.weights = Tensor(np.random.normal(0, sigma, size=(x_dims, 1)))
        self.b = Tensor(np.zeros(1))
        self.lr = lr
    def forward(self, X):
        p = X@self.weights + self.b
        return p
    def loss(self, prediction, label) -> Tensor:
        l = (prediction - label) ** 2 * 0.5
        return l
    def zero_grad(self):
        self.weights.zero_grad()
        self.b.zero_grad()
    def optim(self):
        self.weights.data -= self.lr * self.weights.grad
        self.b.data -= self.lr * self.b.grad

# w = np.array([2, -3.4])
# b = 4.2
# ds = syntheticRegressionData(w, b, (2,), 1000)
ds = carPriceDataset()
model = Model(ds.features, 0.03)

# X shape = (# samples, # features)
# w shape = (# features, 1,)
# X @ w = (# samples, 1)

losses = []
for epoch in range(5000):
    loss = 0.0
    for x, y in ds.stochastic_minibatcher(50):
        model.zero_grad()
        prediction = model.forward(x)
        J = model.loss(prediction, y).mean()
        J.backward()
        loss += J.data.mean()
        model.optim()
    losses.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}\tLoss={loss:.2f}")

fig, axes = plt.subplots(2)

y_hat = model.forward(ds.X)
axes[0].scatter(ds.y.data.reshape(-1), y_hat.data.reshape(-1), s=1)
axes[0].set(xlabel='Actual', ylabel='Predicted')

axes[1].plot([i for i in range(len(losses))], losses)
axes[1].set(xlabel='Epoch', ylabel='Loss')

plt.show()
