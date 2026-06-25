import torch.nn as nn
import torch
import math
from dataclasses import dataclass
from torch.nn import functional as F

@dataclass
class Config:
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    block_size: int
    head_size: int

class AttentionHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.key = nn.Linear(config.embed_dim, config.head_size, bias=False)
        self.query = nn.Linear(config.embed_dim, config.head_size, bias=False)
        self.value = nn.Linear(config.embed_dim, config.head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        att = (q @ k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
        att = att.masked_fill(~self.mask[:T, :T], float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        out = att @ v
        return out

# class CausalSelfAttention(nn.Module):
#     def __init__(self, config: Config):
#         super().__init__()
#         self.head = nn.ModuleList([AttentionHead(config) for _ in range(config.num_heads)])
#         self.proj = nn.Linear(config.num_heads * config.head_size, config.embed_dim, bias=False)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, x):
#         head_outputs = [head(x) for head in self.head]
#         concat = torch.cat(head_outputs, dim=-1)
#         concat = self.proj(concat)
#         concat = self.dropout(concat)
#         return concat

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Cheaper than LayerNorm: no mean-centering and no bias term, only a
    learnable per-feature gain. Used in pre-norm position (modern style).
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute in float32 for numerical stability, then cast back.
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (norm.to(dtype)) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.kqv = nn.Linear(config.embed_dim, 3 * config.head_size * config.num_heads, bias=False)
        self.proj = nn.Linear(config.head_size * config.num_heads, config.embed_dim, bias=False)
        # Flag the residual projection so init can scale it down (GPT-2 style).
        self.proj.RESIDUAL_SCALE = True
        self.dropout = nn.Dropout(0.1)
        self.config = config

    def forward(self, x):
        B, T, C = x.shape
        kqv = self.kqv(x)
        k, q, v = kqv.split(self.config.head_size * self.config.num_heads, dim=-1)
        # q is (B, T, num_heads * head_size)
        k = k.view(B, T, self.config.num_heads, self.config.head_size).transpose(1, 2)
        q = q.view(B, T, self.config.num_heads, self.config.head_size).transpose(1, 2)
        v = v.view(B, T, self.config.num_heads, self.config.head_size).transpose(1, 2)
        # q is (B, num_heads, T, head_size)
        y = F.scaled_dot_product_attention(q, k, v,
         dropout_p=0.1 if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.config.num_heads * self.config.head_size)
        y = self.proj(y)
        y = self.dropout(y)
        return y


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020).

    out = down( SiLU(gate(x)) * up(x) ). Uses a gating branch instead of a
    plain activation. The hidden dim is set to ~8/3 * embed_dim so the total
    parameter count matches a classic 4x GELU MLP despite the extra matrix.
    """
    def __init__(self, config: Config):
        super().__init__()
        hidden = int(8 * config.embed_dim / 3)
        # Round up to a multiple of 64 for efficient matmuls on GPU.
        hidden = 64 * ((hidden + 63) // 64)
        self.gate = nn.Linear(config.embed_dim, hidden, bias=False)
        self.up = nn.Linear(config.embed_dim, hidden, bias=False)
        self.down = nn.Linear(hidden, config.embed_dim, bias=False)
        # Flag the residual projection so init can scale it down.
        self.down.RESIDUAL_SCALE = True
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.down(F.silu(self.gate(x)) * self.up(x))
        x = self.dropout(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ln1 = RMSNorm(config.embed_dim)
        self.mlp = SwiGLU(config)
        self.ln2 = RMSNorm(config.embed_dim)

    def forward(self, x):
        # Pre-norm: normalize the input to each sub-block, keep the residual clean.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class DecoderTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_embedding = nn.Embedding(config.block_size, config.embed_dim)
        self.layers = nn.ModuleList([AttentionBlock(config) for _ in range(config.num_layers)])
        self.ln_f = RMSNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.dropout = nn.Dropout(0.1)

        # Tie weights between token embedding and output head
        self.head.weight = self.token_embedding.weight

        # Explicit, scale-aware initialization.
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * num_layers) so the variance
        # added back into the residual stream stays ~constant with depth.
        residual_std = 0.02 / math.sqrt(2 * config.num_layers)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=residual_std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def forward(self, x):
        batch_size, seq_len = x.shape
        token_emb = self.token_embedding(x)
        pos_emb = self.positional_embedding(torch.arange(seq_len, device=x.device))
        x = token_emb + pos_emb
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        return self

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=100, temperature=1.0, top_k=50):
        """Autoregressive sampling with temperature and top-k filtering.

        - temperature: <1.0 sharpens the distribution, >1.0 flattens it.
          temperature == 0 falls back to greedy (argmax) decoding.
        - top_k: keep only the k most likely tokens before sampling
          (None or <=0 disables the cutoff).
        """
        self.eval()
        input_ids = prompt
        for _ in range(max_new_tokens):
            logits = self(input_ids[:, -self.positional_embedding.num_embeddings:])
            next_token_logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                if top_k is not None and top_k > 0:
                    k = min(top_k, next_token_logits.size(-1))
                    values, _ = torch.topk(next_token_logits, k, dim=-1)
                    threshold = values[:, [-1]]
                    next_token_logits = next_token_logits.masked_fill(
                        next_token_logits < threshold, float('-inf'))
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token_id], dim=1)
        return input_ids[0]
