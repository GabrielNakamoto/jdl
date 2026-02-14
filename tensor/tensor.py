from __future__ import annotations
from enum import Enum, auto
from typing import Union, List
import numpy as np

class Op(Enum):
    CREATE = auto(); POW = auto(); MEAN = auto(); SUM = auto();  EXP = auto()
    ADD = auto(); SUB = auto(); MUL = auto(); DOT = auto(); RESHAPE = auto()
    DIV = auto(); LOG = auto()

class Tensor:
    def __init__(self, data: Union[np.ndarray, list], parents=[], op=Op.CREATE, requires_grad=True):
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        if isinstance(data, np.ndarray):
            data = data.astype(np.float32, copy=False)
        self.requires_grad = requires_grad
        self._sum_axis = None
        self._sum_keepdims = False
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
        out._sum_axis = axis
        out._sum_keepdims = keepdims
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

    @staticmethod
    def _unbroadcast_grad(grad, parent):
        """Reduce grad to match parent.data.shape, undoing NumPy broadcasting."""
        if grad.ndim > parent.data.ndim:
            grad = grad.sum(axis=tuple(range(grad.ndim - parent.data.ndim)))
        axes = tuple(i for i, s in enumerate(parent.data.shape) if s == 1 and grad.shape[i] != 1)
        if axes:
            grad = grad.sum(axis=axes, keepdims=True)
        return grad

    def backward(self):
        topo = self._toposort()
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            if not node.requires_grad: continue
            if node.op == Op.ADD:
                node.parents[0].grad += Tensor._unbroadcast_grad(node.grad, node.parents[0])
                node.parents[1].grad += Tensor._unbroadcast_grad(node.grad, node.parents[1])
            elif node.op == Op.SUB:
                node.parents[0].grad += Tensor._unbroadcast_grad(node.grad, node.parents[0])
                node.parents[1].grad -= Tensor._unbroadcast_grad(node.grad, node.parents[1])
            elif node.op == Op.MUL:
                g0 = node.grad * node.parents[1].data
                g1 = node.grad * node.parents[0].data
                node.parents[0].grad += Tensor._unbroadcast_grad(g0, node.parents[0])
                node.parents[1].grad += Tensor._unbroadcast_grad(g1, node.parents[1])
            elif node.op == Op.DOT:
                node.parents[0].grad += node.grad @ node.parents[1].data.T
                node.parents[1].grad += node.parents[0].data.T @ node.grad
            elif node.op == Op.POW:
                power = node.parents[1].data
                node.parents[0].grad += power * node.parents[0].data ** (power - 1) * node.grad
            elif node.op == Op.MEAN:
                node.parents[0].grad += node.grad / node.parents[0].data.size
            elif node.op == Op.SUM:
                if node._sum_keepdims: grad = node.grad
                else: grad = np.expand_dims(node.grad, axis=node._sum_axis)
                node.parents[0].grad += np.broadcast_to(grad, node.parents[0].data.shape)
            elif node.op == Op.EXP:
                node.parents[0].grad += node.grad * node.data
            elif node.op == Op.DIV:
                a, b = node.parents[0], node.parents[1]
                g0 = node.grad / b.data
                g1 = -node.grad * a.data / (b.data ** 2)
                a.grad += Tensor._unbroadcast_grad(g0, a)
                b.grad += Tensor._unbroadcast_grad(g1, b)
            elif node.op == Op.LOG:
                node.parents[0].grad += node.grad / node.parents[0].data
