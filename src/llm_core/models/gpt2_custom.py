from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_core.parallel.tp_linear import ColumnParallelLinear, RowParallelLinear
from llm_core.modules.base_module import LanguageModelModule, OptimizerConfig


@dataclass
class GPT2CustomConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    max_seq_len: int = 1024
    dropout: float = 0.0
    gradient_checkpointing: bool = True
    tp_size: int = 1


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, tp_size: int) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.tp_size = tp_size
        assert n_head % max(tp_size, 1) == 0
        self.local_heads = n_head // max(tp_size, 1)
        self.local_hidden = self.local_heads * self.head_dim

        self.c_attn = ColumnParallelLinear(n_embd, 3 * n_embd, bias=True, tp_size=tp_size)
        self.c_proj = RowParallelLinear(self.local_hidden, n_embd, bias=True, tp_size=tp_size)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)  # [B, T, 3*local_hidden]
        (q, k, v) = qkv.split(self.local_hidden, dim=2)
        # [B, T, local_heads, head_dim]
        q = q.view(B, T, self.local_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, self.local_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        v = v.view(B, T, self.local_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]

        # Use PyTorch SDPA with causal mask
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.local_hidden)
        y = self.c_proj(y)  # all-reduce inside
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float, tp_size: int) -> None:
        super().__init__()
        hidden = 4 * n_embd
        self.tp_size = tp_size
        self.c_fc = ColumnParallelLinear(n_embd, hidden, bias=True, tp_size=tp_size)
        self.c_proj = RowParallelLinear(hidden // max(tp_size, 1), n_embd, bias=True, tp_size=tp_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.c_proj(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, tp_size: int, use_ckpt: bool = False) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, tp_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout, tp_size)
        self.use_ckpt = use_ckpt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_ckpt:
            x = x + torch.utils.checkpoint.checkpoint(self.attn, self.ln_1(x), use_reentrant=False)
            x = x + torch.utils.checkpoint.checkpoint(self.mlp, self.ln_2(x), use_reentrant=False)
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return x


class GPT2CustomModel(nn.Module):
    def __init__(self, cfg: GPT2CustomConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.max_seq_len, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList([Block(cfg.n_embd, cfg.n_head, cfg.dropout, cfg.tp_size, use_ckpt=cfg.gradient_checkpointing) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        tok = self.wte(input_ids)
        pos_emb = self.wpe(pos)[None, :, :]
        x = self.drop(tok + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


class GPT2CustomLightningModule(LanguageModelModule):
    def __init__(self, cfg: GPT2CustomConfig, optimizer_cfg: OptimizerConfig):
        model = GPT2CustomModel(cfg)
        super().__init__(model=model, tokenizer_vocab_size=cfg.vocab_size, optimizer_config=optimizer_cfg)
