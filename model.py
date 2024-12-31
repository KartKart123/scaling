import torch
import torch.nn as nn
from torch.nn import functional as F
from flash_attn.flash_attn_interface import flash_attn_func

class RotaryEmbedding(nn.Module):
    def __init__(self, max_seq_len, head_dim, rope_theta=10000.0, device=None):
        super().__init__()
        self.max_seq_len_cached = max_seq_len
        self.original_max_seq_len = max_seq_len

        inv_freq, self.attention_scaling = compute_default_rope_parameters(rope_theta, head_dim, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def compute_default_rope_parameters(rope_theta, head_dim, device=None):
        base = rope_theta
        partial_rotary_factor = 1.0
        dim = int(head_dim * partial_rotary_factor)
        attention_factor = 1.0

        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        return inv_freq, attention_factor

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    unsqueeze_dim (`int`, *optional*, defaults to 1):
        The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
        sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
        that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
        k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
        cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
        the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

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

    def forward(self, x, pos_emb):
        B, T, d_model = x.shape

        # Compute Q, K, V
        q = self.query(x).view(B, T, self.n_heads, self.head_dim)  # (B, T, n_heads, head_dim)
        k = self.key(x).view(B, T, self.n_heads, self.head_dim)
        v = self.value(x).view(B, T, self.n_heads, self.head_dim)

        # Apply Rotary Positional Embedding
        cos, sin = pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        # FlashAttention expects (B, T, n_heads, head_dim)
        out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)

        # Flatten back to (B, T, d_model)
        out = out.view(B, T, d_model).contiguous()

        # Final linear
        out = self.proj(out)
        return out
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, intermediate_size=None):
        super().__init__()
        if intermediate_size is None:
            self.intermediate_size = d_model * 4
        else:
            self.intermediate_size = intermediate_size
        self.ln_1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln_2 = RMSNorm(d_model)
        
        # SwiGLU
        self.gate_proj = nn.Linear(d_model, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, d_model, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x, pos_emb):
        # Self-attention + residual
        residual = x
        x = self.ln_1(x)
        x = residual + self.attn(x, pos_emb)

        # SwiGLU + residual
        residual = x
        x = self.ln_2(x)
        x = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        x = residual + x
        return x

class Model(nn.Module):
    def __init__(self, vocab_size=50257, d_model=128, n_heads=4, n_layers=4, max_seq_len=256):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embedding & positional embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.rotary_emb = RotaryEmbedding(max_seq_len, d_model//n_heads)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DecoderLayer(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
        # Final RMSNorm
        self.ln_f = RMSNorm(d_model)
        
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

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.max_seq_len
        
        # Token embeddings
        x = self.token_emb(idx)  # (B, T, d_model)
        
        # Position embeddings
        positions = torch.arange(0, T, device=idx.device).unsqueeze(0)
        pos_emb = self.rotary_emb(x, positions)
        
        # Pass through each Transformer block
        for block in self.blocks:
            x = block(x, pos_emb)
        
        # Final normalization + output projection
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            loss = None

        return logits, loss