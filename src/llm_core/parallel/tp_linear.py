from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from .process_groups import get_tensor_parallel_group


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, tp_size: int = 1) -> None:
        super().__init__()
        self.world_size = max(int(tp_size), 1)
        assert out_features % self.world_size == 0, "out_features must be divisible by tp world size"
        self.out_per_partition = out_features // self.world_size
        self.weight = nn.Parameter(torch.empty(self.out_per_partition, in_features))
        self.bias = nn.Parameter(torch.empty(self.out_per_partition)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / fan_in ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            y = y + self.bias
        return y


class RowParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, tp_size: int = 1) -> None:
        super().__init__()
        self.world_size = max(int(tp_size), 1)
        assert in_features % self.world_size == 0, "in_features must be divisible by tp world size"
        self.in_per_partition = in_features // self.world_size
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_partition))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / fan_in ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x_partial: torch.Tensor) -> torch.Tensor:
        # All-reduce sum of partial outputs across TP group
        y_partial = torch.matmul(x_partial, self.weight.t())
        tp_group = get_tensor_parallel_group()
        if tp_group is not None and dist.is_initialized():
            dist.all_reduce(y_partial, op=dist.ReduceOp.SUM, group=tp_group)
        if self.bias is not None:
            y_partial = y_partial + self.bias
        return y_partial
