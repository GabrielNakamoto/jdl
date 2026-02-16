from __future__ import annotations
from typing import Union
import numpy as np

class Tensor:
    def __init__(self, data: Union[np.ndarray, list], parents=(), requires_grad=True, local_grads=()):
        if isinstance(data, list): data = np.array(data, dtype=np.float32)
        if isinstance(data, np.ndarray): data = data.astype(np.float32, copy=False)
        self.data = data
        self.grad = None

        self._requires_grad = requires_grad
        self._parents = parents
        self._local_grads = local_grads

    def zero_grad(self):
        if self.grad is not None: self.grad.fill(0)

    def one_hot(self, classes):
        samples, = self.shape
        hot = np.zeros((samples,classes))
        hot[np.arange(samples), self.data.astype(dtype=np.int32)]=1
        self.data = hot
        return self
    
    # https://jmlr.org/papers/v15/srivastava14a.html
    def dropout(self, p=0.5):
        mask = (np.random.uniform(size=self.shape) > p).astype(np.float32) / (1.0 - p)
        return Tensor(self.data * mask, parents=(self,), local_grads=(lambda g: g * mask,))

    def __pow__(self, scalar):
        cached = self.data ** (scalar-1)
        return Tensor(cached*self.data, parents=(self,), local_grads=(lambda g: g*scalar*cached,))
    def log(self):
        clamped = np.clip(self.data, 1e-12, None)
        return Tensor(np.log(clamped), parents=(self,), local_grads=(lambda g: g/clamped,))
    def exp(self):
        cached = np.exp(self.data)
        return Tensor(cached, parents=(self,), local_grads=(lambda g: g*cached,))

    def __add__(self, other): return Tensor(self.data + other.data, parents=(self, other), local_grads=(lambda g:g, lambda g:g))
    def __mul__(self, other: Union[Tensor, float]):
        ist = isinstance(other, Tensor)
        factor = other if not ist else other.data
        parents = (self,) if not ist else (self,other)
        return Tensor(self.data * factor, parents=parents, local_grads=(lambda g: g*factor, lambda g: g*self.data))

    def __matmul__(self, other): return Tensor(self.data @ other.data, parents=[self, other], local_grads=(lambda g: g @ other.data.T, lambda g: self.data.T @ g))

    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1.0
    def __truediv__(self, other): return self * (other**-1)
    def __rtruediv__(self, other): return other * (self**-1)

    def sum(self, axis): return Tensor(self.data.sum(axis=axis, keepdims=True), parents=(self,), local_grads=(lambda g: np.ones(self.data.shape) * g,))
    def reshape(self, shape): return Tensor(self.data.reshape(*shape), parents=(self,), local_grads=(lambda g: g.reshape(self.data.shape),))
    def mean(self): return Tensor(self.data.mean(), parents=(self,), local_grads=(lambda g: np.ones(self.data.shape) * g / self.data.size,))

    def relu(self): return Tensor(np.maximum(self.data, 0), parents=(self,), local_grads=(lambda g: g * np.where(self.data > 0, 1, 0),))
    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        return Tensor(s, parents=(self,), local_grads=(lambda g: g * s * (1 - s),))
    def tanh(self):
        e = np.exp(-2 * self.data)
        t = (1 - e) * (1 + e)
        tt = t*t
        return Tensor(t, parents=(self,), local_grads=(lambda g: g * (1 - tt),))

    def softmax(self):# subtract max to prevent overflow, softmax has shift invariance
        e = (self - Tensor(self.data.max(axis=1, keepdims=True))).exp()
        return e / e.sum(axis=1)

    @property
    def shape(self): return self.data.shape

    def _toposort(self):
        topo, visited = [], set()
        def dfs(node):
            if node in visited or not node._requires_grad: return
            visited.add(node)
            for n in node._parents: dfs(n)
            topo.append(node)
        dfs(self)
        return topo

    def backward(self):
        topo = self._toposort()
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node.compute_grad()

    def compute_grad(self):
        for i, p in enumerate(self._parents):
            try: gradient = self._local_grads[i]
            except IndexError: continue
            g = Tensor._unbroadcast_grad(gradient(self.grad), p)
            if p.grad is None: p.grad = g
            else: p.grad += g

    @staticmethod
    def _unbroadcast_grad(grad, parent):
        target = parent.data.shape
        # sum away extra dims
        if grad.ndim > len(target): grad = grad.sum(axis=tuple(range(grad.ndim - len(target))))
        # sum broadcasted gradient axes where parent has size of 1
        # ex. gradient = (3,4) parent = (1,4). Sum along axis=0
        axes = tuple(i for i, s in enumerate(target) if s == 1 and grad.shape[i] != 1)
        if axes: grad = grad.sum(axis=axes, keepdims=True)
        return grad
