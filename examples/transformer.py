import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from jdl import Tensor
from jdl.nn import Linear, LayerNorm, Embedding

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

class FeedForward:
    def __init__(self, dim, hidden_dim):
        self.c_fc = Linear(dim, hidden_dim)
        self.c_proj = Linear(hidden_dim, dim)
    def __call__(self, x):
        return self.c_proj(self.c_fc(x).relu())

class TransformerBlock:
    def __init__(self, dim, n_heads):
        self.attn = MultiHeadAttention(dim, n_heads)
        self.ff = FeedForward(dim, 4*dim)
        self.attn_norm = LayerNorm(dim)
        self.ff_norm = LayerNorm(dim)
    def __call__(self, x):
        h = x + self.attn(self.attn_norm(x))     # Residual normalized attn 
        return h + self.ff(self.ff_norm(h))    # Residual normalized positionwise-FF

class Transformer:
    def __init__(self, dim, n_heads, n_layers, vocab_size, max_context=1024):
        self.tk_emb = Embedding(vocab_size, dim)
        self.pos_emb = Embedding(max_context, dim)
        self.layers = [TransformerBlock(dim, n_heads) for _ in range(n_layers)]
        self.f_norm = LayerNorm(dim)
        self.f_linear = Linear(dim, vocab_size)
        self.allpos = Tensor(np.arange(0, max_context)).reshape((1, -1))
    def __call__(self, tokens): # (batch_size, seq_len)
        bchsz, seqln = tokens.shape
        tok_emb = self.tk_emb(tokens)
        pos = self.allpos[:, :seqln]
        h = tok_emb + self.pos_emb(pos)
        for l in self.layers: h = l(h)

        logits = self.f_linear(self.f_norm(h))
        return logits
