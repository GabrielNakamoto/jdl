from __future__ import annotations
from typing import Union, Tuple
import numpy as np
from numpy.lib.stride_tricks import as_strided

class Tensor:
    def __init__(self, data: Union[np.ndarray, list], parents: Tuple[Tensor, ...]=(), requires_grad=True, local_grads=()):
        if isinstance(data, list): data = np.array(data, dtype=np.float32)
        if isinstance(data, np.ndarray): data = data.astype(np.float32, copy=False)
        self.data, self.grad = data, None

        self._requires_grad = requires_grad
        self._parents = parents
        self._local_grads = local_grads

    @property
    def shape(self): return self.data.shape
    @staticmethod
    def scalar(s): return Tensor(np.array(s))

    def zero_grad(self):
        if self.grad is not None: self.grad.fill(0)

    # --- Unary Ops --
    def log(self):
        clamped = np.clip(self.data, 1e-12, None)
        return Tensor(np.log(clamped), parents=(self,), local_grads=(lambda g: g/clamped,))
    def exp(self):
        cached = np.exp(self.data)
        return Tensor(cached, parents=(self,), local_grads=(lambda g: g*cached,))
    def __pow__(self, scalar):
        cached = self.data ** (scalar-1)
        return Tensor(cached*self.data, parents=(self,), local_grads=(lambda g: g*scalar*cached,))
    def sqrt(self): return self ** 0.5
    def normalize(self, mean, var, eps = 1e-6):
        eps = Tensor(np.array(eps), requires_grad=False)
        return (self - mean) / (var + eps).sqrt()

    # --- Binary Ops ---
    def __add__(self, other: Union[Tensor, float]):
        ist = isinstance(other, Tensor)
        factor = other if not ist else other.data
        return Tensor(self.data + factor, parents=(self,other) if ist else (self,), local_grads=(lambda g:g, lambda g:g))
    def __mul__(self, other: Union[Tensor, float]):
        ist = isinstance(other, Tensor)
        factor = other if not ist else other.data
        return Tensor(self.data * factor, parents=(self,other) if ist else (self,), local_grads=(lambda g: g*factor, lambda g: g*self.data))
    def __matmul__(self, other): return Tensor(self.data @ other.data, parents=[self, other], local_grads=(lambda g: g @ other.data.swapaxes(-2, -1), lambda g: self.data.swapaxes(-2, -1) @ g))

    # --- Composed/Reverse Ops ---
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1.0
    def __truediv__(self, other): return self * (other**-1)
    def __rtruediv__(self, other): return other * (self**-1)

    # --- Data Access ---
    def __getitem__(self, idx):
        def grad(g):
            grad = np.zeros_like(self.data)
            np.add.at(grad, idx, g)
            return grad
        return Tensor(self.data[idx], parents=(self,), local_grads=(grad,))

    # --- Reshape Ops ---
    def flatten(self, start=0): return self.reshape(self.shape[:start] + (-1,))
    def reshape(self, shape): return Tensor(self.data.reshape(*shape), parents=(self,), local_grads=(lambda g: g.reshape(self.data.shape),))
    def transpose(self, dim0=1, dim1=0): return Tensor(self.data.swapaxes(dim0, dim1), parents=(self,), local_grads=(lambda g: g.swapaxes(dim0, dim1),))

    # --- Reduce Ops ---
    def sum(self, axis=None): return Tensor(self.data.sum(axis=axis, keepdims=True), parents=(self,), local_grads=(lambda g: np.ones(self.data.shape) * g,))
    def mean(self, axis=None):
        if axis is None: n = self.data.size
        elif isinstance(axis, tuple): n = np.prod([self.data.shape[a] for a in axis])
        else: n = self.data.shape[axis]
        return self.sum(axis=axis) * (1.0 / n)
    def max(self, axis=None):
        out = self.data.max(axis=axis, keepdims=True)
        mask = (self.data == out).astype(np.float32)
        mask /= mask.sum(axis=axis, keepdims=True)
        return Tensor(out, parents=(self,), local_grads=(lambda g: g*mask,))

    # --- Activation Functions ---
    def relu(self): return Tensor(np.maximum(self.data, 0), parents=(self,), local_grads=(lambda g: g * np.where(self.data > 0, 1, 0),))
    # https://arxiv.org/pdf/1606.08415
    def gelu(self):
        # Fused GELU: single forward pass, custom backward (avoids ~8 intermediate Tensors)
        x = self.data
        c = np.sqrt(2.0 / np.pi)
        inner = c * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        out = 0.5 * x * (1 + t)
        def grad(g):
            sech2 = 1 - t ** 2  # derivative of tanh
            d_inner = c * (1 + 0.134145 * x ** 2)  # 0.134145 = 3 * 0.044715
            return g * (0.5 * (1 + t) + 0.5 * x * d_inner * sech2)
        return Tensor(out, parents=(self,), local_grads=(grad,))
    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        return Tensor(s, parents=(self,), local_grads=(lambda g: g * s * (1 - s),))
    def tanh(self):
        e = np.exp(-2 * self.data)
        t = (1 - e) / (1 + e)
        tt = t*t
        return Tensor(t, parents=(self,), local_grads=(lambda g: g * (1 - tt),))

    # --- General NN ---
    @staticmethod
    def he_init(shape):
        return Tensor(np.random.normal(0, np.sqrt(2.0/shape[0]), shape))
    def one_hot(self, classes):
        orig_shape = self.shape
        flat = self.data.flatten().astype(np.int32)
        hot = np.zeros((flat.size, classes))
        hot[np.arange(flat.size), flat]=1
        return Tensor(hot.reshape((*orig_shape, classes)))
    def softmax(self, axis=-1):# subtract max to prevent overflow, softmax has shift invariance
        e = (self - Tensor(self.data.max(axis=axis, keepdims=True))).exp()
        return e / e.sum(axis=axis)
    def log_softmax(self):
        m = Tensor(self.data.max(axis=-1, keepdims=True))
        shifted = self - m
        sl =  (shifted).exp().sum(axis=-1).log()
        return shifted - sl
    def sparse_categorical_crossentropy(self, y):
        return -((y.one_hot(self.shape[-1]) * self.log_softmax()).sum(axis=-1).mean())
    def dropout(self, p=0.5, training=True):
        # Paper: https://jmlr.org/papers/v15/srivastava14a.html
        if not training:
            return self
        mask = (np.random.uniform(size=self.shape) > p).astype(np.float32) / (1.0 - p)
        return Tensor(self.data * mask, parents=(self,), local_grads=(lambda g: g * mask,))
    # https://arxiv.org/pdf/1706.03762v7
    def scaled_dot_product_attention(self, k, v, causal_mask=False): # (q, k, v) <- (batch, n_heads, seq, head_dim)
        seq_len = self.shape[2]
        scale = Tensor(np.array(1.0 / np.sqrt(k.shape[-1])), requires_grad=False)
        scores = (self @ k.transpose(2, 3)) * scale # (batch, n_heads, seq, seq)
        if causal_mask: # Dont allow tokens to 'look' at future tokens
            mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
            scores = scores + Tensor(mask, requires_grad=False)
        attn = scores.softmax() # (batch, n_heads, seq, seq)
        return attn @ v # (batch, n_heads, seq, head_dim)

    # --- CNN/Pooling ---
    def im2col(self, kshape, stride, padding):
        # References:
        # - https://rileylearning.medium.com/convolution-layer-with-numpy-5d8cca3c2152
        # - https://medium.com/@sundarramanp2000/different-implementations-of-the-ubiquitous-convolution-6a9269dbe77f
        # 
        # Parameters:
        # - 4d image tensor with shape NHWC
        # - shape of kernel / window
        # - kerne stride
        # - image padding
        # 
        # - Let a window be a unique 2d range over which the kernel is applied
        # - This transformation takes each unique window, respecting strides, padding 
        #   and input/output channels, and flattens them into collumns
        # 
        # - Outputs a 2d collumn tensor with shape (# of windows, size of a window)
        N, h, w, c = self.shape
        kh, kw, = kshape[:2]
        oh, ow = (h - kh + 2*padding[0]) // stride[0] + 1, (w - kw + 2*padding[1]) // stride[1] + 1

        img = np.pad(self.data, [(0,0),(padding[0],padding[0]),(padding[1],padding[1]),(0,0)])
        shape = (N, oh, ow, kh, kh, c)

        strides = (
            img.strides[0], # N: next batch image
            stride[0] * img.strides[1], # oh: jump conv stride (vert)
            stride[1] * img.strides[2], # ow: jump conv stride (hor)
            img.strides[1], # kh: move 1 row within window
            img.strides[2], # kw: move 1 col within window
            img.strides[3], # c: move 1 channel
        )
        # extracts input patches (windows kernel is applied to) and stores as output collumns
        cols = as_strided(img, shape=shape, strides=strides).copy().reshape(N * oh * ow, c * kh * kw)

        def backward(g):
            dcols = g.reshape(N, oh, ow, kh, kw, c)
            dimg = np.zeros_like(img)
            for y in range(kh):
                y_max = y + stride[0]*oh
                for x in range(kw):
                    x_max = x + stride[1]*ow
                    dimg[:,y:y_max:stride[0],x:x_max:stride[1],:] += dcols[:,:,:,y,x,:]
            ph, pw = padding
            return dimg[:, ph:ph+h, pw:pw+w, :]

        return Tensor(cols, parents=(self,), local_grads=(backward,)), oh, ow

    def conv2d(self, k, stride: int | Tuple[int, ...], padding: int | Tuple[int, ...]):
        if isinstance(padding, int): padding = (padding,padding)
        if isinstance(stride, int): stride = (stride,stride)
        kh, kw, ci, co = k.shape
        cols, oh, ow = self.im2col(k.shape, stride, padding)
        k_flat = k.reshape((kh * kw * ci, co))
        return (cols @ k_flat).reshape((self.shape[0], oh, ow, co))

    def _pool2d(self, pool_shape, stride: None | int | Tuple[int, ...], padding: int | Tuple[int, ...], func):
        if stride is None: stride = pool_shape
        if isinstance(padding, int): padding = (padding,padding)
        if isinstance(stride, int): stride = (stride,stride)
        N, c = self.shape[0], self.shape[-1]
        ph, pw = pool_shape
        cols, oh, ow = self.im2col(pool_shape, stride, padding)
        return func(cols.reshape((N*oh*ow, ph*pw, c)), axis=1).reshape((N, oh, ow, c))

    def avg_pool2d(self, pool_shape, stride: None | int | Tuple[int, ...]=None, padding: int | Tuple[int, ...]=0):
        return self._pool2d(pool_shape, stride, padding, Tensor.mean)
    def max_pool2d(self, pool_shape, stride: None | int | Tuple[int, ...]=None, padding: int | Tuple[int, ...]=0):
        return self._pool2d(pool_shape, stride, padding, Tensor.max)


    # --- Autograd Engine ---
    def backward(self):
        topo = self._toposort()
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._compute_grad()
        return self

    def _compute_grad(self):
        for i, p in enumerate(self._parents):
            try: gradient = self._local_grads[i]
            except IndexError: continue
            g = Tensor._unbroadcast_grad(gradient(self.grad), p)
            if p.grad is None: p.grad = g
            else: p.grad += g

    def _toposort(self):
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if not node._requires_grad: continue
            if processed:
                topo.append(node)
                continue
            if node in visited: continue
            visited.add(node)
            stack.append((node, True))
            for p in node._parents:
                stack.append((p, False))
        return topo

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
