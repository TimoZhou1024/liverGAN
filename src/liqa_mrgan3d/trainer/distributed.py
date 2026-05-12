"""Helpers for multi-GPU FSDP training.

Runs under ``torchrun --nproc_per_node=N scripts/train.py --config …``. In
single-GPU mode (no torchrun env vars) these helpers degrade to no-ops so the
same trainer code works on a laptop and on an 8×3090 server without branching.

>>> NOTE: the FSDP path requires ≥2 visible CUDA devices on a machine with
>>> torchrun. Your local box has 1× 3090, so only the single-GPU path is
>>> exercised here. Run the multi-GPU smoke on the remote 8×3090 server.
"""
from __future__ import annotations

import functools
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import (
    CPUOffload,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


def init_distributed() -> tuple[bool, int, int]:
    """Initialise torch.distributed if torchrun env vars are present.

    Returns ``(is_dist, rank, world_size)``. Calling this in a single-GPU run
    is a no-op and returns ``(False, 0, 1)``.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size <= 1:
        return False, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, world_size


def _mixed_precision_policy(use_amp: bool) -> MixedPrecision | None:
    if not use_amp:
        return None
    return MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )


def fsdp_wrap(
    model: nn.Module,
    *,
    min_params: int = 10_000_000,
    sharding: ShardingStrategy = ShardingStrategy.FULL_SHARD,
    checkpoint: bool = True,
    checkpoint_target_cls: tuple[type, ...] = (),
    use_amp: bool = True,
    cpu_offload: bool = False,
) -> nn.Module:
    """Wrap a module with FSDP and optional activation checkpointing.

    * ``sharding`` controls whether params/grads/optimizer state are sharded.
      Use ``FULL_SHARD`` for the big networks (G, UNet) and ``SHARD_GRAD_OP``
      for the smaller Reg/Discriminators so their forward params stay local
      (faster) but grads + optimizer state still shard.
    * ``checkpoint=True`` wraps instances of ``checkpoint_target_cls`` with
      activation checkpointing, trading compute for activation memory.
    """
    if checkpoint and checkpoint_target_cls:
        # Imported lazily — this path is only executed under distributed.
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=lambda submodule: isinstance(submodule, checkpoint_target_cls),
        )

    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_params)
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding,
        mixed_precision=_mixed_precision_policy(use_amp),
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        use_orig_params=True,
    )


def gather_full_state_dict(module: nn.Module) -> dict | None:
    """Gather a FULL_STATE_DICT on rank 0 for checkpoint saving.

    Returns ``None`` on non-zero ranks. On plain (non-FSDP) modules simply
    returns the ordinary state_dict on rank 0.
    """
    rank = int(os.environ.get("RANK", "0"))
    if not isinstance(module, FSDP):
        return module.state_dict() if rank == 0 else None
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, cfg):
        state = module.state_dict()
    return state if rank == 0 else None


def barrier() -> None:
    """No-op when not distributed."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
