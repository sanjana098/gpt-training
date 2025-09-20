from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 2000
    max_steps: int = 100000


class LanguageModelModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        tokenizer_vocab_size: int,
        optimizer_config: OptimizerConfig,
        precision: str = "bf16",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])  # keep config, not raw model
        self.tokenizer_vocab_size = tokenizer_vocab_size
        self.optimizer_config = optimizer_config
        self.precision_choice = precision
        self.label_smoothing = label_smoothing
        self.loss_fct = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]  # [B, T]
        logits = self.forward(input_ids)  # [B, T, V]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        logits = self.forward(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": loss}

    def configure_optimizers(self):
        decay_params, nodecay_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or "ln" in name or "layernorm" in name.lower():
                nodecay_params.append(param)
            else:
                decay_params.append(param)
        param_groups = [
            {"params": decay_params, "weight_decay": self.optimizer_config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(param_groups, lr=self.optimizer_config.learning_rate, betas=self.optimizer_config.betas, eps=self.optimizer_config.eps)

        def lr_lambda(step: int) -> float:
            warmup = self.optimizer_config.warmup_steps
            max_steps = max(self.optimizer_config.max_steps, 1)
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, max_steps - warmup))
            # Cosine decay to 10%
            return 0.1 + 0.9 * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)))

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
