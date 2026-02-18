from __future__ import annotations
from typing import Union, Tuple
import numpy as np

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

    # --- Binary Ops ---
    def __add__(self, other): return Tensor(self.data + other.data, parents=(self, other), local_grads=(lambda g:g, lambda g:g))
    def __mul__(self, other: Union[Tensor, float]):
        ist = isinstance(other, Tensor)
        factor = other if not ist else other.data
        return Tensor(self.data * factor, parents=(self,other) if ist else (self,), local_grads=(lambda g: g*factor, lambda g: g*self.data))
    def __matmul__(self, other): return Tensor(self.data @ other.data, parents=[self, other], local_grads=(lambda g: g @ other.data.T, lambda g: self.data.T @ g))

    # --- Composed/Reverse Ops ---
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1.0
    def __truediv__(self, other): return self * (other**-1)
    def __rtruediv__(self, other): return other * (self**-1)

    # --- Reshape Ops ---
    def flatten(self, start=0): return self.reshape(self.shape[:start] + (-1,))
    def reshape(self, shape): return Tensor(self.data.reshape(*shape), parents=(self,), local_grads=(lambda g: g.reshape(self.data.shape),))

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
    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        return Tensor(s, parents=(self,), local_grads=(lambda g: g * s * (1 - s),))
    def tanh(self):
        e = np.exp(-2 * self.data)
        t = (1 - e) / (1 + e)
        tt = t*t
        return Tensor(t, parents=(self,), local_grads=(lambda g: g * (1 - tt),))

    # --- General NN ---
    def one_hot(self, classes):
        samples, = self.shape
        hot = np.zeros((samples,classes))
        hot[np.arange(samples), self.data.astype(dtype=np.int32)]=1
        return Tensor(hot)
    def softmax(self):# subtract max to prevent overflow, softmax has shift invariance
        e = (self - Tensor(self.data.max(axis=1, keepdims=True))).exp()
        return e / e.sum(axis=1)
    def log_softmax(self):
        m = Tensor(self.data.max(axis=1, keepdims=True))
        shifted = self - m
        sl =  (shifted).exp().sum(axis=1).log()
        return shifted - sl
    def sparse_categorical_crossentropy(self, y):
        return -((y.one_hot(self.shape[1]) * self.log_softmax()).sum(axis=1).mean())
    def dropout(self, p=0.5):
        # Paper: https://jmlr.org/papers/v15/srivastava14a.html
        mask = (np.random.uniform(size=self.shape) > p).astype(np.float32) / (1.0 - p)
        return Tensor(self.data * mask, parents=(self,), local_grads=(lambda g: g * mask,))

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
        cols = np.zeros((N,oh,ow,kh,kw,c))

        for y in range(kh):
            y_max = y + stride[0] * oh
            for x in range(kw):
                x_max = x + stride[1] * ow
                cols[:, :, :, y, x, :]+=img[:, y:y_max:stride[0], x:x_max:stride[1], :]

        # extracts input patches (windows kernel is applied to) and stores as output collumns
        out = cols.reshape(N * oh * ow, c * kh * kw)

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

        return Tensor(out, parents=(self,), local_grads=(backward,)), oh, ow

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
        def dfs(node):
            if node in visited or not node._requires_grad: return
            visited.add(node)
            for n in node._parents: dfs(n)
            topo.append(node)
        dfs(self)
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
