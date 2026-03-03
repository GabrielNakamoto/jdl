import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from jdl import Tensor
from jdl.nn import Linear, LayerNorm, Embedding, MultiHeadAttention, ADAM

class FeedForward:
    def __init__(self, dim, hidden_dim):
        self.c_fc = Linear(dim, hidden_dim)
        self.c_proj = Linear(hidden_dim, dim)
    def params(self): return *self.c_fc.params(), *self.c_proj.params()
    def __call__(self, x):
        return self.c_proj(self.c_fc(x).relu())

class TransformerBlock:
    def __init__(self, dim, n_heads):
        self.attn = MultiHeadAttention(dim, n_heads)
        self.ff = FeedForward(dim, 4*dim)
        self.attn_norm = LayerNorm(dim)
        self.ff_norm = LayerNorm(dim)
    def params(self): return *self.attn_norm.params(), *self.ff_norm.params(), *self.attn.params(), *self.ff.params()
    def __call__(self, x, causal=False):
        h = x + self.attn(self.attn_norm(x), causal=causal)     # Residual normalized attn  (batch, seqln, dims)
        return h + self.ff(self.ff_norm(h))    # Residual normalized positionwise-FF


# Transformer Decoder
class NanoGPT:
    def __init__(self, vocab_size, dim=128, n_heads=4, n_layers=4, max_context=512):
        self.tok_emb = Embedding(vocab_size, dim)
        self.pos_emb = Embedding(max_context, dim)
        self.blocks = [TransformerBlock(dim, n_heads) for _ in range(n_layers)]
        self.ln_f = LayerNorm(dim)
        self.lm_head = Linear(dim, vocab_size)
        self.max_context = max_context
    def __call__(self, tokens): # (batch_size, seq_len)
        bchsz, seq_len = tokens.shape
        # Embed tokens + positions
        tok_emb = self.tok_emb(tokens)
        pos_emb = self.pos_emb(range(seq_len))
        x = tok_emb + pos_emb

        # Transformer blocks
        for block in self.blocks: x = block(x, causal=True)

        # Project to vocabulary
        logits = self.lm_head(self.ln_f(x)) # (batch, seq_len, vocab_size)
        return logits
    def generate(self, tokens, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            context = tokens[:, -self.max_context:]
            logits = self(context)
            logits = logits[:, -1, :] / temperature
            probs = logits.softmax().data
            next_token = np.random.choice(len(probs[0]), p=probs[0])
            tokens = np.concatenate([tokens, [[next_token]]], axis=1)
        return tokens

num_steps = 100
text = open("datasets/dostoevsky.txt").read()
chars = sorted(set(text))
vocab_size = len(chars)

char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
encoder = lambda s: [char_to_idx[c] for c in s]
decoder = lambda t: ''.join(idx_to_char[i] for i in t)

num_steps = 1000
print_every = 100

model = NanoGPT(vocab_size)
optimizer = ADAM(model, step_size=3e-4)

data = np.array(encoder(text))

def get_batch(batch_size, context_len):
    starts = np.random.randint(0, len(data) - context_len, size=batch_size)
    x = np.stack([data[i:i+context_len] for i in starts])
    y = np.stack([data[i+1:i+context_len+1] for i in starts])
    return x, Tensor(y)

for step in range(num_steps):
    optimizer.zero()
    x, y = get_batch(batch_size=32, context_len=64)
    logits = model(x)
    l = logits.sparse_categorical_crossentropy(y).backward()
    optimizer.step()

    if step % print_every == 0:
        print(f"step {step:5d} | loss {l.flatten().mean().data[0]}")
        prompt = np.array([[char_to_idx['\n']]])
        generated = model.generate(prompt, max_new_tokens=100)
        print(decoder(generated.flatten()))
        #print(f"Sample: {decoder(generated[0])[:100]}")
