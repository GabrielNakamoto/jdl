from enum import Enum, auto
from typing import Union, List
import numpy as np

class Op(Enum):
    CREATE = auto()
    ADD = auto()
    POW = auto()
    MEAN = auto()
    SUB = auto()
    MUL = auto()
    DOT = auto()
    RESHAPE = auto()


class Tensor:
    def __init__(self, data: Union[np.ndarray, list], parents=[], op=Op.CREATE):
        if isinstance(data, list):
            data = np.array(data)
        self.data = data
        self.parents = parents
        self.op = op
        self.grad = np.zeros_like(data)

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def mean(self):
        return Tensor(self.data.mean(), parents=[self], op=Op.MEAN)
    def __pow__(self, scalar):
        return Tensor(self.data ** scalar, parents=[self, Tensor(scalar)], op=Op.POW)
    def __add__(self, other):
        return Tensor(self.data + other.data, parents=[self, other], op=Op.ADD)
    def __sub__(self, other):
        return Tensor(self.data - other.data, parents=[self,other], op=Op.SUB)
    def __mul__(self, scalar):
        return Tensor(self.data * scalar, parents=[self, Tensor(scalar)], op=Op.MUL)
    def __matmul__(self, other):
        return Tensor(self.data @ other.data, parents=[self, other], op=Op.DOT)
    def reshape(self, shape):
        return Tensor(self.data.reshape(*shape), parents=[self], op=Op.RESHAPE)

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

    # Ensure shape of grad = shape of parent
    # Only way this wouldnt be true is from broadcasting from previous backpass op
    @staticmethod
    def _unbroadcast(node, parent):
        grad = node.grad
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
            if node.op == Op.ADD:
                node.parents[0].grad += Tensor._unbroadcast(node, node.parents[0])
                node.parents[1].grad += Tensor._unbroadcast(node, node.parents[1])
            elif node.op == Op.SUB:
                node.parents[0].grad += Tensor._unbroadcast(node, node.parents[0])
                node.parents[1].grad -= Tensor._unbroadcast(node, node.parents[1])
            elif node.op == Op.MUL:
                node.parents[0].grad += node.grad * node.parents[1].data
            elif node.op == Op.DOT:
                node.parents[0].grad += node.grad @ node.parents[1].data.T
                node.parents[1].grad += node.parents[0].data.T @ node.grad
            elif node.op == Op.POW:
                power = node.parents[1].data
                node.parents[0].grad += power * node.parents[0].data ** (power - 1) * node.grad
            elif node.op == Op.MEAN:
                node.parents[0].grad += node.grad * np.ones_like(node.parents[0].data) / node.parents[0].data.size
