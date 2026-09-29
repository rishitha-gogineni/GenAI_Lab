import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, context_length):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(context_length, context_length))
        self.register_buffer("mask", mask.view(1, 1, context_length, context_length))

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.out_drop(self.out(y))


class FeedForward(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_size, dropout, context_length):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, num_heads, dropout, context_length)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, ffn_size, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        m = config["model"]
        context = config["data"]["context_length"]
        hidden = m["hidden_size"]
        self.context_length = context
        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(context, hidden)
        self.drop = nn.Dropout(m["dropout"])
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden, m["num_heads"], m["ffn_size"], m["dropout"], context)
            for _ in range(m["num_layers"])
        ])
        self.ln_f = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_chars, temperature=1.0):
        for _ in range(max_new_chars):
            x = idx[:, -self.context_length:]
            logits, _ = self(x)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
