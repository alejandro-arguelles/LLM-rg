import torch.nn as nn
import torch
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
        self.key = nn.Linear(config.embed_dim, config.head_size)
        self.query = nn.Linear(config.embed_dim, config.head_size)
        self.value = nn.Linear(config.embed_dim, config.head_size)
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

class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.head = nn.ModuleList([AttentionHead(config) for _ in range(config.num_heads)])
        self.proj = nn.Linear(config.num_heads * config.head_size, config.embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        head_outputs = [head(x) for head in self.head]
        concat = torch.cat(head_outputs, dim=-1)
        concat = self.proj(concat)
        concat = self.dropout(concat)
        return concat

class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.kqv = nn.Linear(config.embed_dim, 3 * config.head_size * config.num_heads)
        self.proj = nn.Linear(config.head_size * config.num_heads, config.embed_dim)
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


class AttentionBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * config.embed_dim, config.embed_dim)
        )
        self.ln2 = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class DecoderTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_embedding = nn.Embedding(config.block_size, config.embed_dim)
        self.layers = nn.ModuleList([AttentionBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size)
        self.dropout = nn.Dropout(0.1)

        # Tie weights between token embedding and output head
        self.head.weight = self.token_embedding.weight
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        token_emb = self.token_embedding(x)
        pos_emb = self.positional_embedding(torch.arange(seq_len, device=x.device))
        x = token_emb + pos_emb
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        x = self.dropout(x)
        logits = self.head(x)
        return logits
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        return self
    
    def generate(self, prompt, encoder, decoder, max_new_tokens=100):
        self.eval()
        input_ids = torch.tensor([encoder[c] for c in prompt], dtype=torch.long).unsqueeze(0).to("cuda")
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self(input_ids[:, -self.positional_embedding.num_embeddings:])
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.multinomial(torch.nn.functional.softmax(next_token_logits, dim=-1), num_samples=1)
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
        generated_text = ''.join([decoder[ix.item()] for ix in input_ids[0]])
        return generated_text
