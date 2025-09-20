from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from ..modules.base_module import LanguageModelModule, OptimizerConfig


@dataclass
class GPT2LightningConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    max_seq_len: int = 1024
    dropout: float = 0.0
    gradient_checkpointing: bool = True


class GPT2LightningModule(LanguageModelModule):
    def __init__(self, cfg: GPT2LightningConfig, optimizer_cfg: OptimizerConfig):
        gpt2_cfg = GPT2Config(
            vocab_size=cfg.vocab_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            n_positions=cfg.max_seq_len,
            n_ctx=cfg.max_seq_len,
            resid_pdrop=cfg.dropout,
            embd_pdrop=cfg.dropout,
            attn_pdrop=cfg.dropout,
            bos_token_id=50256,
            eos_token_id=50256,
        )
        model = GPT2LMHeadModel(gpt2_cfg)
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        # Adjust forward to return logits only for base module
        class Wrapper(nn.Module):
            def __init__(self, m: GPT2LMHeadModel):
                super().__init__()
                self.m = m
            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                out = self.m(input_ids=input_ids, use_cache=False)
                return out.logits
        super().__init__(model=Wrapper(model), tokenizer_vocab_size=cfg.vocab_size, optimizer_config=optimizer_cfg)
