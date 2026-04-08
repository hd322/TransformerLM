import torch
import torch.nn as nn
from collections.abc import Iterable

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        my_std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0, std=my_std, a=-3*my_std, b = 3*my_std)

    def forward(self, x):
        return x @ self.weight.T
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)
    def forward(self, token_ids):
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype) 
        )
    
    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        divide = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        rms = (x / divide) * self.weight
        return rms.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        silu = self.w1(x)
        silu = silu * torch.sigmoid(silu) # [..., d_ff]
        glu = self.w3(x) #[..., d_ff]
        glu = silu * glu # [..., d_ff]

        return self.w2(glu)
    
def softmax(in_features, dim):

    max_val = torch.max(in_features, dim=dim, keepdim=True)[0]
    exp = torch.exp(in_features - max_val)

    base = torch.sum(exp, dim=dim, keepdim=True)

    return  exp/base

class RoPE(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        self.d_k = d_k

        k = torch.arange(0, d_k // 2, device=device)
        freqs = 1.0 / (theta ** (2 * k / d_k)) # (d_k //2, 1)

        positions = torch.arange(max_seq_len, device=device) #（max_len, 1）
        angles = torch.outer(positions, freqs) # (max_len, d_k //2)

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x, token_positions):
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        out1 = x1 * cos - x2 * sin # (token_pos, d_k //2)
        out2 = x1 * sin + x2 * cos # (token_pos, d_k //2)

        out = torch.stack([out1, out2], dim=-1) # (token_pos, d_k //2, 2)
        return out.flatten(-2)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, theta=None, max_seq_len=None, device=None, dtype=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query = Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.key = Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.value = Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.wo = Linear(embed_dim, embed_dim,device=device, dtype=dtype)
        # self.rope = RoPE(theta, embed_dim, max_seq_len)

        self.head_dim = embed_dim // num_heads

        if theta is not None and theta > 0:
            self.rope = RoPE(theta, self.head_dim, max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, x, token_positions=None):
        *batch, seq_len, embded_dim = x.shape

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # (..., len_seq, embed_dim) -> (..., embed_dim, len_seq) -> -> (..., num_head, len_seq, head_dim)
        query = query.view(*batch, seq_len, self.num_heads, self.head_dim).transpose(-2, -3)
        key = key.view(*batch, seq_len, self.num_heads, self.head_dim).transpose(-2, -3)
        value = value.view(*batch, seq_len, self.num_heads, self.head_dim).transpose(-2, -3)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)

            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)

        QK = torch.matmul(query, key.transpose(-1, -2)) # (..., num_head, len_seq, len_seq)
        QK = QK / (self.head_dim ** 0.5)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        QK = QK.masked_fill(mask==0, float('-inf'))
        A = softmax(QK, dim=-1)

        output = torch.matmul(A, value) # (..., num_head, len_seq, head_dim)

        output = output.transpose(-2, -3).contiguous().view(*batch, seq_len, embded_dim)

        return self.wo(output)
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, theta=None, max_seq_len=None, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(embed_dim, device=device, dtype=dtype)
        self.mha = MultiHeadAttention(embed_dim, num_heads,theta, max_seq_len, device, dtype)
        self.ln2 = RMSNorm(embed_dim, device=device, dtype=dtype)
        self.ffn = SwiGLU(embed_dim, ffn_dim, device=device, dtype=dtype)

    def forward(self, x, token_positions):

        shortcut = x
        output = self.ln1(x)
        output = self.mha(output, token_positions=token_positions)
        output += shortcut

        shortcut = output
        output = self.ln2(output)
        output = self.ffn(output)

        return output+shortcut

def _manual_clip_grad_norm(parameters, max_l2_norm):
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return
    
    device = grads[0].device
    total_norm_sq = torch.tensor(0.0, device=device)
    
    for g in grads:
        total_norm_sq += torch.sum(g.detach() ** 2)
        
    total_norm = torch.sqrt(total_norm_sq)

    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for g in grads:
            g.detach().mul_(clip_coef)


def save_checkpoint(model, optimizer, iteration, out_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out_path)
    
def load_checkpoint(src_path, model, optimizer):
    checkpoint = torch.load(src_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['iteration']






