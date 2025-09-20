import os
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.strategies import FSDPStrategy

from llm_core.parallel.process_groups import init_tensor_parallel_groups


@dataclass
class TPConfig:
    tp_size: int


class TP_FSDP_Strategy(FSDPStrategy):
    def __init__(self, cfg) -> None:
        fs = cfg.fsdp
        super().__init__(
            use_orig_params=fs.use_orig_params,
            activation_checkpointing=fs.activation_checkpointing,
            cpu_offload=fs.cpu_offload,
        )
        self.tp_size = int(cfg.parallel.tp_size)

    def setup_environment(self) -> None:
        super().setup_environment()
        init_tensor_parallel_groups(self.tp_size)

    def teardown(self) -> None:
        # Let base clean up
        super().teardown()
