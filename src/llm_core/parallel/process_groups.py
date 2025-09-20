import os
from typing import List, Optional, Tuple

import torch.distributed as dist


_tp_group = None  # type: ignore
_tp_group_world_size = 1
_tp_group_rank = 0


def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _compute_node_and_local(rank: int, local_world_size: int) -> Tuple[int, int]:
    node_index = rank // local_world_size
    local_rank = rank % local_world_size
    return node_index, local_rank


def init_tensor_parallel_groups(tp_size: int) -> None:
    """Create tensor-parallel process groups confined within a node.

    Assumes torchrun assigns contiguous ranks per node and sets LOCAL_WORLD_SIZE.
    """
    global _tp_group, _tp_group_world_size, _tp_group_rank

    if not dist.is_available() or not dist.is_initialized() or tp_size <= 1:
        _tp_group = None
        _tp_group_world_size = 1
        _tp_group_rank = 0
        return

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_world_size = _get_int_env("LOCAL_WORLD_SIZE", world_size)
    if local_world_size % tp_size != 0:
        raise ValueError(f"LOCAL_WORLD_SIZE={local_world_size} not divisible by tp_size={tp_size}")

    num_nodes = world_size // local_world_size
    node_index, local_rank = _compute_node_and_local(rank, local_world_size)

    # Build all TP groups across all nodes
    groups: List[List[int]] = []
    groups_per_node = local_world_size // tp_size
    for n in range(num_nodes):
        for g in range(groups_per_node):
            start = g * tp_size
            members = [n * local_world_size + start + i for i in range(tp_size)]
            groups.append(members)

    my_group: Optional[List[int]] = None
    # Which intra-node group does this local rank belong to?
    my_group_id_within_node = local_rank // tp_size
    start = my_group_id_within_node * tp_size
    my_group = [node_index * local_world_size + start + i for i in range(tp_size)]

    # Create a new group for every list to ensure all processes call in the same order
    created_groups = []
    for members in groups:
        g = dist.new_group(ranks=members)
        created_groups.append(g)
        if rank in members:
            _tp_group = g
            _tp_group_world_size = tp_size
            _tp_group_rank = members.index(rank)


def get_tensor_parallel_group():
    return _tp_group


def get_tensor_parallel_world_size() -> int:
    return _tp_group_world_size


def get_tensor_parallel_rank() -> int:
    return _tp_group_rank
