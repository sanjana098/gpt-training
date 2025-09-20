from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy, DeepSpeedStrategy


def make_strategy(name: str, precision: str, cfg) -> pl.strategies.Strategy:
    if name == "ddp":
        return DDPStrategy(find_unused_parameters=False)
    if name == "fsdp":
        fs = cfg.fsdp
        return FSDPStrategy(
            auto_wrap_policy=None,
            use_orig_params=fs.use_orig_params,
            activation_checkpointing=fs.activation_checkpointing,
            cpu_offload=fs.cpu_offload,
        )
    if name.startswith("deepspeed_zero"):
        zero_stage = 2 if name.endswith("2") else 3
        ds_cfg = {
            "train_micro_batch_size_per_gpu": 1,  # PL manages batch size
            "gradient_accumulation_steps": 1,
            "zero_optimization": {
                "stage": zero_stage,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_bucket_size": 5e7,
                "stage3_max_live_parameters": 1e9,
                "stage3_prefetch_bucket_size": 5e8,
                "stage3_param_persistence_threshold": 1e6,
            },
            "bf16": {"enabled": precision.startswith("bf16")},
            "fp16": {"enabled": precision.startswith("16")},
        }
        return DeepSpeedStrategy(config=ds_cfg)
    if name == "tp_fsdp":
        from .tp_fsdp_strategy import TP_FSDP_Strategy
        return TP_FSDP_Strategy(cfg)
    raise ValueError(f"Unknown strategy {name}")
