import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm
from jdl import Tensor
from jdl.nn import Linear, LayerNorm, Embedding, MultiHeadAttention, ADAM

class FeedForward:
    def __init__(self, dim, hidden_dim):
        self.c_fc = Linear(dim, hidden_dim)
        self.c_proj = Linear(hidden_dim, dim)
    def params(self): return *self.c_fc.params(), *self.c_proj.params()
    def __call__(self, x):
        return self.c_proj(self.c_fc(x).gelu())

class TransformerBlock:
    def __init__(self, dim, n_heads):
        self.attn = MultiHeadAttention(dim, n_heads)
        self.ff = FeedForward(dim, 4*dim)
        self.attn_norm = LayerNorm(dim)
        self.ff_norm = LayerNorm(dim)
    def params(self): return *self.attn_norm.params(), *self.ff_norm.params(), *self.attn.params(), *self.ff.params()
    def __call__(self, x, causal=False, training=True):
        h = x + self.attn(self.attn_norm(x), causal=causal, training=training).dropout(0.1, training=training)     # Residual normalized attn  (batch, seqln, dims)
        return h + self.ff(self.ff_norm(h)).dropout(0.1, training=training)    # Residual normalized positionwise-FF


# Transformer Decoder

# Why is this considered just a decoder and not an encoder?
# - pure self attention
# - normal architecture;
# 
class NanoGPT:
    def __init__(self, vocab_size, dim, n_heads, n_layers, max_context=512):
        self.tok_emb = Embedding(vocab_size, dim)
        self.pos_emb = Embedding(max_context, dim)
        self.blocks = [TransformerBlock(dim, n_heads) for _ in range(n_layers)]
        self.ln_f = LayerNorm(dim)
        self.lm_head = Linear(dim, vocab_size)
        self.max_context = max_context
    def __call__(self, tokens, training=True): # (batch_size, seq_len)
        bchsz, seq_len = tokens.shape
        # Embed tokens + positions
        tok_emb = self.tok_emb(tokens)
        pos_emb = self.pos_emb(range(seq_len))
        x = (tok_emb + pos_emb).dropout(0.1, training=training)

        # Transformer blocks
        for block in self.blocks: x = block(x, causal=True, training=training)

        # Project to vocabulary
        logits = self.lm_head(self.ln_f(x)) # (batch, seq_len, vocab_size)
        return logits
    def generate(self, tokens, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            context = tokens[:, -self.max_context:]
            logits = self(context, training=False)
            logits = logits[:, -1, :] / temperature
            probs = logits.softmax().data
            next_token = np.random.choice(len(probs[0]), p=probs[0])
            tokens = np.concatenate([tokens, [[next_token]]], axis=1)
        return tokens

num_steps = 100
text = open("datasets/dostoevsky.txt").read()
chars = sorted(set(text))
vocab_size = len(chars)

print("Vocab:", chars)

char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
encoder = lambda s: [char_to_idx[c] for c in s]
decoder = lambda t: ''.join(idx_to_char[i] for i in t)

num_steps = 10000
print_every = 100

model = NanoGPT(vocab_size, 384, 8, 8, 256)
optimizer = ADAM(model, step_size=3e-4)

data = np.array(encoder(text))

def get_batch(batch_size, context_len):
    starts = np.random.randint(0, len(data) - context_len, size=batch_size)
    x = np.stack([data[i:i+context_len] for i in starts])
    y = np.stack([data[i+1:i+context_len+1] for i in starts])
    return x, Tensor(y)

def get_lr(step, warmup_steps=100, max_steps=5000, max_lr=3e-4, min_lr=1e-5):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * decay_ratio))

for step in tqdm(range(num_steps), desc="Training"):
    optimizer.step_size = get_lr(step)
    optimizer.zero()
    x, y = get_batch(batch_size=32, context_len=256)
    logits = model(x)
    l = logits.sparse_categorical_crossentropy(y).backward()
    optimizer.step()

    if step % print_every == 0:
        tqdm.write(f"step {step:5d} | loss {l.flatten().mean().data[0]:.4f}")
        prompt = np.array([[char_to_idx['\n']]])
        generated = model.generate(prompt, max_new_tokens=100)
        tqdm.write(f"Sample: {decoder(generated.flatten())}")
