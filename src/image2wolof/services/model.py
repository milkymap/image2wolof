"""Six-layer pre-norm transformer decoder with tied embeddings, cross-attending to projected SigLIP2 patch tokens."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .dataset import BOS, EOS, MAX_LENGTH, PAD

LEGACY = {"memory_layers": 0, "add_memory_positions": False}  # options of the research code, inactive in these checkpoints


def sinusoid(d: int, length: int = MAX_LENGTH) -> torch.Tensor:
    pos = torch.arange(length)[:, None]
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000) / d))
    pe = torch.zeros(length, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size: int, d: int = 512, layers: int = 6, heads: int = 8, ff: int = 2048,
                 dropout: float = 0.1, feature_dim: int = 768, grid_size: int = 14):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, d=d, layers=layers, heads=heads, ff=ff, dropout=dropout,
                           feature_dim=feature_dim, grid_size=grid_size)
        self.d, self.dropout, self.memory_shape = d, dropout, (grid_size ** 2, feature_dim)
        self.bridge = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, d), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(d, d), nn.LayerNorm(d))
        self.embed = nn.Embedding(vocab_size, d, padding_idx=PAD)
        nn.init.normal_(self.embed.weight, std=d ** -0.5)
        with torch.no_grad(): self.embed.weight[PAD].zero_()
        self.register_buffer("positions", sinusoid(d), persistent=False)
        layer = nn.TransformerDecoderLayer(d, heads, ff, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(d))
        for block in self.decoder.layers:
            for name, p in block.named_parameters():
                if p.ndim > 1: nn.init.xavier_uniform_(p)
                elif "norm" in name and name.endswith("weight"): nn.init.ones_(p)
                else: nn.init.zeros_(p)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    @classmethod
    def from_checkpoint(cls, ck: dict) -> "CaptionDecoder":
        """Build from a checkpoint (research or image2wolof format) and load its weights strictly."""
        config = dict(ck["config"])
        for k, neutral in LEGACY.items():
            if config.pop(k, neutral) != neutral: raise ValueError(f"unsupported checkpoint option {k}")
        model = cls(**config)
        model.load_state_dict(ck["model"])
        return model

    def memory(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or tuple(features.shape[1:]) != self.memory_shape:
            raise ValueError(f"unexpected visual feature shape {tuple(features.shape)}")
        return self.bridge(features.float())

    def decode(self, tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens) * math.sqrt(self.d)
        x = F.dropout(x + self.positions[:tokens.shape[1]].to(x.dtype), self.dropout, self.training)
        mask = torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool).triu(1)
        return self.head(self.decoder(x, memory, tgt_mask=mask, tgt_key_padding_mask=tokens.eq(PAD)))

    def forward(self, features: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return self.decode(tokens, self.memory(features))

    @torch.no_grad()
    def generate(self, features: torch.Tensor, max_tokens: int = 96) -> torch.Tensor:
        """Greedy decoding from BOS; finished rows are padded."""
        memory = self.memory(features)
        tokens = torch.full((features.shape[0], 1), BOS, dtype=torch.long, device=features.device)
        done = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
        for _ in range(max_tokens):
            logits = self.decode(tokens, memory)[:, -1].float()
            logits[:, :EOS] = -torch.inf
            nxt = logits.argmax(-1).masked_fill(done, PAD)
            tokens = torch.cat((tokens, nxt[:, None]), 1)
            done |= nxt.eq(EOS)
            if done.all(): break
        return tokens


def objective(model: CaptionDecoder, features: torch.Tensor, tokens: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """Teacher-forced cross-entropy with label smoothing, padding ignored."""
    logits = model(features, tokens[:, :-1])
    return F.cross_entropy(logits.flatten(0, 1).float(), tokens[:, 1:].reshape(-1), ignore_index=PAD, label_smoothing=smoothing)
