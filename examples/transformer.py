import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jdl import Tensor
from jdl.nn import Linear

class MultiHeadAttention:
    def __init__(self, dim, n_heads):
        self.fused_qkv = Linear(dim, dim*3)
        self.proj_out = Linear(dim, dim)
        self.dim = dim
        self.n_heads = n_heads
        self.head_dims = dim // n_heads
    def __call__(self, x): # (batch, seq len, dim)
        # 1. Project input vectors into (q, k, v ) vectors based on ff layer
        # 2. Reshape each vector into sub-heads to allow more complex attention learning
        xqkv = self.fused_qkv(x).reshape((x.shape[0], x.shape[1], 3, self.n_heads, self.head_dims)) # (batch, seq, 3, heads, head_dim)
        # 3. Reshape q, k, v E (batch, n_heads, seq, head_dim) for dot product attention
        q = xqkv[:, :, 0, :, :].transpose(1, 2)
        k = xqkv[:, :, 1, :, :].transpose(1, 2)
        v = xqkv[:, :, 2, :, :].transpose(1, 2)

        # 4. Compute dot product attention and reshape back
        attn = q.scaled_dot_product_attention(k, v).transpose(1, 2) # (batch, seq, n_heads, head_dim)
        # 5. Concatenate heads and project back to single dimension from (attn_q, attn_k, attn_v) -> x
        return self.proj_out(attn.reshape(x.shape[0], x.shape[1], self.dim)) # (batch, seq, dim)

class TransformerBlock:
    def __init__(self):
        self.attn = MultiHeadAttention(0,0)
        pass
    def __call__(self, x):
        pass
