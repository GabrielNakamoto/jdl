from .tensor import Tensor
from .ops import Op
import numpy as np

def compute_grad(node: Tensor):
    try: grad_map[node.op](node)
    finally: return

def _unbroadcast_grad(grad, parent):
    """Reduce grad to match parent.data.shape, undoing NumPy broadcasting."""
    if grad.ndim > parent.data.ndim:
        grad = grad.sum(axis=tuple(range(grad.ndim - parent.data.ndim)))
    axes = tuple(i for i, s in enumerate(parent.data.shape) if s == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad

def add_grad(node: Tensor):
    node.parents[0].grad += _unbroadcast_grad(node.grad, node.parents[0])
    node.parents[1].grad += _unbroadcast_grad(node.grad, node.parents[1])

def sub_grad(node: Tensor):
    node.parents[0].grad += _unbroadcast_grad(node.grad, node.parents[0])
    node.parents[1].grad -= _unbroadcast_grad(node.grad, node.parents[1])

def mul_grad(node: Tensor):
    g0 = node.grad * node.parents[1].data
    g1 = node.grad * node.parents[0].data
    node.parents[0].grad += _unbroadcast_grad(g0, node.parents[0])
    node.parents[1].grad += _unbroadcast_grad(g1, node.parents[1])

def div_grad(node: Tensor):
    a, b = node.parents[0], node.parents[1]
    g0 = node.grad / b.data
    g1 = -node.grad * a.data / (b.data ** 2)
    a.grad += _unbroadcast_grad(g0, a)
    b.grad += _unbroadcast_grad(g1, b)

def dot_grad(node: Tensor):
    node.parents[0].grad += node.grad @ node.parents[1].data.T
    node.parents[1].grad += node.parents[0].data.T @ node.grad

def pow_grad(node: Tensor):
    power = node.parents[1].data
    node.parents[0].grad += power * node.parents[0].data ** (power - 1) * node.grad

def mean_grad(node: Tensor):
    node.parents[0].grad += node.grad / node.parents[0].data.size

def sum_grad(node: Tensor):
    if node.sum_keepdims: grad = node.grad
    else: grad = np.expand_dims(node.grad, axis=node.sum_axis)
    node.parents[0].grad += np.broadcast_to(grad, node.parents[0].data.shape)

def exp_grad(node: Tensor):
    node.parents[0].grad += node.grad * node.data

def log_grad(node: Tensor):
    node.parents[0].grad += node.grad / node.parents[0].data

grad_map = {
    Op.ADD : add_grad,
    Op.SUB : sub_grad,
    Op.MUL : mul_grad,
    Op.DIV : div_grad,
    Op.MEAN : mean_grad,
    Op.LOG : log_grad,
    Op.EXP : exp_grad,
    Op.SUM : sum_grad,
    Op.DOT : dot_grad,
    Op.POW : pow_grad
}
