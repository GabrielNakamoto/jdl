from __future__ import annotations
from .ops import Op
from typing import Union, List
import numpy as np

class Tensor:
    def __init__(self, data: Union[np.ndarray, list], parents=[], op=Op.CREATE, requires_grad=True):
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        if isinstance(data, np.ndarray):
            data = data.astype(np.float32, copy=False)
        self.requires_grad = requires_grad
        self.sum_axis = None
        self.sum_keepdims = False
        self.data = data
        self.parents = parents
        self.op = op
        self.grad = np.zeros_like(data, dtype=np.float32)

    def zero_grad(self):
        self.grad.fill(0)

    def mean(self):
        return Tensor(self.data.mean(), parents=[self], op=Op.MEAN)
    def __pow__(self, scalar):
        return Tensor(self.data ** scalar, parents=[self, Tensor(scalar)], op=Op.POW)
    def __add__(self, other):
        return Tensor(self.data + other.data, parents=[self, other], op=Op.ADD)
    def __sub__(self, other):
        return Tensor(self.data - other.data, parents=[self,other], op=Op.SUB)
    def __mul__(self, other: Union[Tensor, float]):
        if isinstance(other, float): return Tensor(self.data * other, parents=[self, Tensor(np.array(other))], op=Op.MUL)
        elif isinstance(other, Tensor):
            return Tensor(self.data * other.data, parents=[self,other], op=Op.MUL)
    def __neg__(self):
        return self * -1.0
    def __truediv__(self, other: Union[Tensor, float]):
        if isinstance(other, float):
            return self * (1.0 / other)
        elif isinstance(other, Tensor):
            return Tensor(self.data / other.data, parents=[self,other], op=Op.DIV)
    def __matmul__(self, other):
        return Tensor(self.data @ other.data, parents=[self, other], op=Op.DOT)
    def sum(self, axis, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), parents=[self], op=Op.SUM)
        out.sum_axis = axis
        out.sum_keepdims = keepdims
        return out
    def log(self):
        return Tensor(np.log(self.data), parents=[self], op=Op.LOG)
    def exp(self):
        return Tensor(np.exp(self.data), parents=[self], op=Op.EXP)
    def reshape(self, shape):
        return Tensor(self.data.reshape(*shape), parents=[self], op=Op.RESHAPE)
    def softmax(self):
        shifted = self - Tensor(self.data.max(axis=1, keepdims=True))
        e = shifted.exp()
        return e / e.sum(axis=1, keepdims=True)

    def _toposort(self):
        topo = []
        visited = set()
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for n in node.parents:
                dfs(n)
            topo.append(node)
        dfs(self)
        return topo

    @property
    def shape(self): return self.data.shape

    def backward(self):
        from . import engine
        topo = self._toposort()
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            if node.requires_grad:
                engine.compute_grad(node)    
