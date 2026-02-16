from .tensor import Tensor
import numpy as np

def compute_grad(node: Tensor):
    for i, p in enumerate(node.parents):
        try: gradient = node.local_grads[i]
        except IndexError: continue
        g = _unbroadcast_grad(gradient(node.grad), p)
        if p.grad is None: p.grad = g
        else: p.grad += g

def _unbroadcast_grad(grad, parent):
    target = parent.data.shape
    # sum away extra dims
    if grad.ndim > len(target):
        grad = grad.sum(axis=tuple(range(grad.ndim - len(target))))
    # sum broadcasted gradient axes where parent has size of 1
    # ex. gradient = (3,4) parent = (1,4). Sum along axis=0
    axes = tuple(i for i, s in enumerate(target) if s == 1 and grad.shape[i] != 1)
    if axes: grad = grad.sum(axis=axes, keepdims=True)
    return grad
