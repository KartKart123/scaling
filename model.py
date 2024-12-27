import torch
import torch.nn as nn
from torch.nn import functional as F
from flash_attn.flash_attn_interface import flash_attn_func

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key   = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape

        # Compute Q, K, V
        q = self.query(x)  # (B, T, d_model)
        k = self.key(x)
        v = self.value(x)

        # Reshape to (B, T, n_heads, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)

        # FlashAttention expects (B, S, H, D), causal=True
        # -> returns output shape (B, S, H, D)
        out = flash_attn_func(q, k, v, 
                              dropout_p=0.0, 
                              softmax_scale=None, 
                              causal=True)

        # Flatten back to (B, T, d_model)
        out = out.view(B, T, C)

        # Final linear
        out = self.proj(out)
        return out
    
class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        
        # A basic feed-forward
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4*d_model, bias=False),
            nn.GELU(),
            nn.Linear(4*d_model, d_model, bias=False),
        )

    def forward(self, x):
        # Self-attention + residual
        x = x + self.attn(self.ln_1(x))
        # Feed-forward + residual
        x = x + self.mlp(self.ln_2(x))
        return x

class Model(nn.Module):
    def __init__(self, vocab_size=50257, d_model=128, n_heads=4, n_layers=4, max_seq_len=256):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embedding & positional embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
        # Final LayerNorm
        self.ln_f = nn.LayerNorm(d_model)
        
        # Linear head to vocab logits
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying 
        self.token_emb.weight = self.head.weight

        self.apply(self._init_weights)

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos_emb.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        """
        idx shape: (B, T) with integer token IDs
        Returns: (B, T, vocab_size) logits
        """
        B, T = idx.size()
        assert T <= self.max_seq_len, f"Cannot forward sequence of length {T}, max sequence size is only {self.max_seq_len}"
        
        # Token + positional embeddings
        tok_emb = self.token_emb(idx)  # (B, T, d_model)
        
        # Positions 0..T-1
        positions = torch.arange(0, T, device=idx.device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)  # (1, T, d_model)
        
        x = tok_emb + pos_emb
        
        # Pass through each Transformer block
        for block in self.blocks:
            x = block(x)
        
        # Final layernorm + output projection
        x = self.ln_f(x)

        if targets is not None:
            logits = self.head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss