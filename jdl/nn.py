from __future__ import annotations
from jdl.tensor import Tensor
import numpy as np
from typing import List, Tuple


def get_model_params(model):
    params = []
    for _, value in vars(model).items():
        if hasattr(value, 'params'): params.extend(value.params())
    return params

# --- NN Layers ---
class RMSNorm:
    def __init__(self, dim):
        self.weight = Tensor(np.ones(dim))
    def params(self): return self.weight,
    def __call__(self, x):
        ms = (x ** 2).mean(axis=-1)
        rms = (ms + 1e-5).sqrt()
        return x / rms * self.weight

class MultiHeadAttention:
    def __init__(self, dim, n_heads):
        self.fused_qkv = Linear(dim, dim*3)
        self.proj = Linear(dim, dim)
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
    def params(self): return *self.fused_qkv.params(), *self.proj.params()
    def __call__(self, x, causal, training=True): # (batch, seq_len, dim)
        bchsz, seqln = x.shape[:2]
        xqkv = self.fused_qkv(x).reshape((bchsz, seqln, 3, self.n_heads, self.head_dim))
        q = xqkv[:, :, 0, :, :].transpose(1, 2)
        k = xqkv[:, :, 1, :, :].transpose(1, 2)
        v = xqkv[:, :, 2, :, :].transpose(1, 2)
        attn = q.scaled_dot_product_attention(k, v, causal_mask=causal, training=training).transpose(2, 1)
        return self.proj(attn.reshape((bchsz, seqln, self.dim)))

class Embedding:
    def __init__(self, vocab_size, embed_size):
        self.weight = Tensor(np.random.normal(0.0, 0.02, (vocab_size, embed_size)))
    def params(self): return self.weight,
    def __call__(self, indices): # indices (batch, seq_len)
        return self.weight[indices]

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size, self.hidden_size = input_size, hidden_size
        gates = ['i', 'f', 'o', 'c']
        self.w_x = {g: Tensor.he_init((input_size, hidden_size)) for g in gates}
        self.w_h = {g: Tensor.he_init((hidden_size, hidden_size)) for g in gates}
        self.b  = {g:Tensor(np.zeros(hidden_size)) for g in gates}
    def params(self): return (*self.w_x.values(), *self.w_h.values(), *self.b.values())
    def __call__(self, x, state=None):
        batch_size = x.data.shape[0]
        if state is None: h, c = Tensor(np.zeros((batch_size, self.hidden_size))), Tensor(np.zeros((batch_size, self.hidden_size)))
        else: h, c = state

        i = (x @ self.w_x["i"] + h @ self.w_h["i"] + self.b["i"]).sigmoid()
        o = (x @ self.w_x["o"] + h @ self.w_h["o"] + self.b["o"]).sigmoid()
        f = (x @ self.w_x["f"] + h @ self.w_h["f"] + self.b["f"]).sigmoid()
        c_tilde = (x @ self.w_x["c"] + h @ self.w_h["c"] + self.b["c"]).tanh()
        c = f * c + i * c_tilde
        h = o * c.tanh()
        return h, (h,c)

class LayerNorm:
    # https://arxiv.org/pdf/1607.06450v1
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.weights = Tensor(np.ones(hidden_size))
        self.bias = Tensor(np.zeros(hidden_size))
    def params(self): return self.weights, self.bias
    def __call__(self, x):
        mean = x.mean(axis=-1)
        var = ((x - mean) ** 2).mean(axis=-1)
        return x.normalize(mean, var) * self.weights + self.bias

class BatchNorm:
    # Paper: https://arxiv.org/pdf/1502.03167v3
    def __init__(self, size):
        self.weights = Tensor(np.ones(size))
        self.bias = Tensor(np.zeros(size))
    def params(self): return self.weights, self.bias
    def __call__(self, x):
        axis = tuple(range(x.data.ndim - 1))
        mean = x.mean(axis=axis)
        var = ((x - mean) ** 2).mean(axis=axis)
        return x.normalize(mean, var) * self.weights + self.bias

class Conv2d:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int,int]=1,
        padding: int | Tuple[int,int]=0
    ):
        if isinstance(kernel_size, int): kernel_size = (kernel_size,kernel_size)
        if isinstance(stride, int): stride = (stride,stride)
        if isinstance(padding, int): padding = (padding,padding)

        self.padding, self.stride = padding, stride
        self.channels = (in_channels, out_channels)

        self.weights = Tensor(np.random.normal(0, np.sqrt(2.0/kernel_size[0]*kernel_size[1]*in_channels), (kernel_size + self.channels)))
        self.bias = Tensor(np.zeros(1))
    def params(self): return self.weights, self.bias
    def __call__(self, x):
        return x.conv2d(self.weights, self.stride, self.padding) + self.bias

class Linear:
    def __init__(self, inputs, outputs):
        self.shape = (inputs,outputs)
        self.weights = Tensor(np.random.normal(0, np.sqrt(2.0/inputs), (inputs,outputs)))
        self.bias = Tensor(np.zeros(outputs))
    def params(self): return self.weights, self.bias
    def __call__(self, x: Tensor):
        orig_shape = x.shape[:-1]
        x = x.reshape((-1, self.shape[0]))
        out = x @ self.weights + self.bias
        return out.reshape((*orig_shape, self.shape[1]))

# --- Optimizers ---
class Optimizer:
    def __init__(self, params: List[Tensor]):
        self.params = params
    def zero(self):
        for p in self.params: p.zero_grad()


class SGD(Optimizer):
    def __init__(self, model, lr=0.01):
        super().__init__(get_model_params(model))
        self.lr = lr
    def step(self):
        for p in self.params: p.data -= self.lr * p.grad

class ADAM(Optimizer):
    # Paper: https://arxiv.org/pdf/1412.6980
    def __init__(self, model, step_size=0.001, decay_rates=(0.9,0.999)):
        super().__init__(get_model_params(model))
        self.step_size, self.t = step_size, 0
        self.b1, self.b2 = decay_rates
        self.m, self.v = [0.0] * len(self.params), [0.0] * len(self.params)
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * p.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * p.grad * p.grad
            m1_hat = self.m[i] / (1 - self.b1 ** self.t)
            v1_hat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.step_size * m1_hat / (np.sqrt(v1_hat) + 1e-8)
