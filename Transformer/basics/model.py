import torch
import torch.nn as nn
from basics.utils import Linear, Embedding, RMSNorm, TransformerBlock

class TransformerLM(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=d_model,
                num_heads=num_heads,
                ffn_dim=d_ff,
                theta=rope_theta,
                max_seq_len=context_length,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head  = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len)  LongTensor
        Returns:
            logits:    (batch, seq_len, vocab_size)
        """
        seq_len = token_ids.shape[1]
        token_positions = torch.arange(seq_len, device=token_ids.device)

        x = self.token_embeddings(token_ids)          # (B, T, d_model)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        x = self.ln_final(x)                          # (B, T, d_model)
        return self.lm_head(x)                         # (B, T, vocab_size)