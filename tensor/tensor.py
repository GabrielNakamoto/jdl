from __future__ import annotations
from typing import Union, List
import numpy as np

class Tensor:
    def __init__(self, data: Union[np.ndarray, list], parents=[], requires_grad=True, local_grads=()):
        if isinstance(data, list): data = np.array(data, dtype=np.float32)
        if isinstance(data, np.ndarray): data = data.astype(np.float32, copy=False)
        self.requires_grad = requires_grad
        self.data = data
        self.parents = parents
        self.grad = None
        self.local_grads = local_grads

    def zero_grad(self):
        if self.grad is not None: self.grad.fill(0)

    def __pow__(self, scalar):
        cached = self.data ** (scalar-1)
        return Tensor(cached*self.data, parents=[self,], local_grads=(lambda g: g*scalar*cached,))
    def log(self): return Tensor(np.log(self.data), parents=[self], local_grads=(lambda g: g/self.data,))
    def exp(self):
        cached = np.exp(self.data)
        return Tensor(cached, parents=[self], local_grads=(lambda g: g*cached,))

    def __add__(self, other): return Tensor(self.data + other.data, parents=[self, other], local_grads=(lambda g:g, lambda g:g))
    def __sub__(self, other): return Tensor(self.data - other.data, parents=[self,other], local_grads=(lambda g:g, lambda g:-g))
    def __mul__(self, other: Union[Tensor, float]):
        factor = other
        parents = (self,)
        if isinstance(other, Tensor):
            factor = other.data
            parents = (self,other)
        return Tensor(self.data * factor, parents=list(parents), local_grads=(lambda g: g*factor, lambda g: g*self.data))

    def __matmul__(self, other): return Tensor(self.data @ other.data, parents=[self, other], local_grads=(lambda g: g @ other.data.T, lambda g: self.data.T @ g))

    def __rsub__(self, other): return self - other
    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1.0
    def __truediv__(self, other: Union[Tensor, float]): return self * (other**-1)

    def sum(self, axis): return Tensor(self.data.sum(axis=axis, keepdims=True), parents=[self], local_grads=(lambda g: np.ones(self.data.shape) * g,))
    def reshape(self, shape): return Tensor(self.data.reshape(*shape), parents=[self], local_grads=(lambda g: g.reshape(self.data.shape),))
    def mean(self): return Tensor(self.data.mean(), parents=[self], local_grads=(lambda g: np.ones(self.data.shape) * g / self.data.size,))

    def softmax(self):
        # subtract max to prevent overflow, softmax has shift invariance
        shifted = self - Tensor(self.data.max(axis=1, keepdims=True))
        e = shifted.exp()
        return e / e.sum(axis=1)

    @property
    def shape(self): return self.data.shape

    def _toposort(self):
        topo = []
        visited = set()
        def dfs(node):
            if node in visited or not node.requires_grad:
                return
            visited.add(node)
            for n in node.parents:
                dfs(n)
            topo.append(node)
        dfs(self)
        return topo

    def backward(self):
        from . import engine
        topo = self._toposort()
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            engine.compute_grad(node)    
