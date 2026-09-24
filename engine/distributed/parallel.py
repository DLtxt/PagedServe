"""Process layout, groups, and the collectives tensor and pipeline parallelism need.

One process per GPU. With tensor-parallel size T and pipeline-parallel size P there are T * P ranks,
laid out stage-major: stage s owns ranks s*T .. s*T + T - 1, so tp rank t of stage s is global rank
s*T + t. Rank 0 is the driver: it runs the scheduler, block manager, tokenizer and HTTP server, and it
is also stage 0's tp rank 0.

Two kinds of traffic:

  data plane     the work itself. Tensor parallelism all-reduces inside every layer and gathers the
                 vocab-split logits; pipeline parallelism sends hidden states from stage to stage.
                 NCCL on GPUs, gloo on CPUs.
  control plane  the driver sending each step's plan to every rank, and the sampled tokens coming
                 back. Always gloo on CPU tensors, so reading a plan never costs a GPU sync.

Plans travel point to point, not by broadcast: a broadcast is a collective every rank enters
together, so the driver could not hand stage 0 its next batch before the last stage had finished the
current one, and a pipeline could never hold two batches. Sends are non-blocking; each kind of
message has its own tag, and every rank receives each kind in the order it was sent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class ParallelState:
    tp_size: int = 1
    pp_size: int = 1
    rank: int = 0
    tp_group: object | None = None  # this stage's tensor-parallel ranks (data plane)
    control_group: object | None = None  # every rank, gloo (control plane)

    @property
    def world_size(self) -> int:
        return self.tp_size * self.pp_size

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def tp_rank(self) -> int:
        return self.rank % self.tp_size

    @property
    def pp_rank(self) -> int:
        return self.rank // self.tp_size

    @property
    def is_driver(self) -> bool:
        return self.rank == 0

    @property
    def is_first_stage(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_last_stage(self) -> bool:
        return self.pp_rank == self.pp_size - 1

    def global_rank(self, pp_rank: int, tp_rank: int) -> int:
        return pp_rank * self.tp_size + tp_rank

    @property
    def sampler_rank(self) -> int:
        """Where full logits exist: tp rank 0 of the last stage, after the vocab gather."""
        return self.global_rank(self.pp_size - 1, 0)


SINGLE = ParallelState()

PLAN_TAG, HIDDEN_TAG, RESULT_TAG = 1, 2, 3


def layer_range(num_layers: int, pp_size: int, pp_rank: int) -> tuple[int, int]:
    """The contiguous block of layers a pipeline stage owns; earlier stages take any remainder."""
    base, extra = divmod(num_layers, pp_size)
    start = pp_rank * base + min(pp_rank, extra)
    return start, start + base + (1 if pp_rank < extra else 0)


def check_parallel(model_cfg, tp_size: int, pp_size: int) -> None:
    """Refuse layouts the sharding cannot express, with the reason."""
    problems = []
    for name, value in (
        ("num_attention_heads", model_cfg.num_attention_heads),
        ("num_key_value_heads", model_cfg.num_key_value_heads),
        ("intermediate_size", model_cfg.intermediate_size),
        ("vocab_size", model_cfg.vocab_size),
    ):
        if value % tp_size:
            problems.append(f"{name}={value} is not divisible by tensor_parallel_size={tp_size}")
    if pp_size > model_cfg.num_hidden_layers:
        problems.append(f"pipeline_parallel_size={pp_size} exceeds the model's {model_cfg.num_hidden_layers} layers")
    if problems:
        raise ValueError("; ".join(problems))


def init_parallel(
    tp_size: int,
    pp_size: int,
    device_type: str,
    rank: int | None = None,
    world_size: int | None = None,
    init_method: str | None = None,
) -> ParallelState:
    """Join the process group. Under torchrun the rank and world size come from the environment;
    spawned test workers pass them explicitly with a tcp:// init_method."""
    rank = int(os.environ["RANK"]) if rank is None else rank
    world_size = int(os.environ["WORLD_SIZE"]) if world_size is None else world_size
    if world_size != tp_size * pp_size:
        raise ValueError(f"world size {world_size} != tensor_parallel_size {tp_size} * pipeline_parallel_size {pp_size}")
    backend = "nccl" if device_type == "cuda" else "gloo"
    if device_type == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    dist.init_process_group(backend, init_method=init_method, rank=rank, world_size=world_size)
    # Every rank creates every group, in the same order, including groups it is not in.
    tp_group = None
    for stage in range(pp_size):
        ranks = [stage * tp_size + t for t in range(tp_size)]
        group = dist.new_group(ranks)
        if rank in ranks:
            tp_group = group
    control = dist.group.WORLD if backend == "gloo" else dist.new_group(backend="gloo")
    return ParallelState(tp_size, pp_size, rank, tp_group, control)


def shutdown_parallel() -> None:
    if not dist.is_initialized():
        return
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        # NCCL sends and receives return before the GPU runs them; a stage that never samples may still
        # have its last hidden-state transfer in flight. Tearing the communicators down under it can hang.
        torch.cuda.synchronize()
    dist.destroy_process_group()


# --- data plane ----------------------------------------------------------------------------------


def tp_all_reduce(ps: ParallelState, x: torch.Tensor) -> torch.Tensor:
    if ps.tp_size > 1:
        dist.all_reduce(x, group=ps.tp_group)
    return x


def tp_gather_last_dim(ps: ParallelState, x: torch.Tensor) -> torch.Tensor | None:
    """Concatenate each tp rank's shard along the last dimension onto tp rank 0 (None elsewhere)."""
    if ps.tp_size == 1:
        return x
    x = x.contiguous()
    parts = [torch.empty_like(x) for _ in range(ps.tp_size)] if ps.tp_rank == 0 else None
    dist.gather(x, parts, dst=ps.global_rank(ps.pp_rank, 0), group=ps.tp_group)
    return torch.cat(parts, dim=-1) if parts is not None else None


def isend_to_next_stage(ps: ParallelState, x: torch.Tensor) -> tuple:
    """Returns (work, tensor): the tensor must stay alive until the work completes."""
    x = x.contiguous()
    return dist.isend(x, dst=ps.global_rank(ps.pp_rank + 1, ps.tp_rank), tag=HIDDEN_TAG), x


def recv_from_prev_stage(ps: ParallelState, shape: tuple[int, ...], dtype: torch.dtype, device) -> torch.Tensor:
    buf = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(buf, src=ps.global_rank(ps.pp_rank - 1, ps.tp_rank), tag=HIDDEN_TAG)
    return buf


# --- control plane -------------------------------------------------------------------------------


def all_ranks_min(ps: ParallelState, value: int) -> int:
    """Every rank must agree on shared sizes (block ids are global): take the smallest."""
    if not ps.distributed:
        return value
    t = torch.tensor([value], dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=ps.control_group)
    return int(t.item())


def send_plan(ps: ParallelState, plan: torch.Tensor) -> list[tuple]:
    """Driver -> every other rank: the plan's length, then the plan (1-D int64, CPU). Returns the
    (work, tensor) pairs, which must stay alive until they complete."""
    length = torch.tensor([plan.numel()], dtype=torch.int64)
    handles = []
    for rank in range(1, ps.world_size):
        handles.append((dist.isend(length, dst=rank, group=ps.control_group, tag=PLAN_TAG), length))
        handles.append((dist.isend(plan, dst=rank, group=ps.control_group, tag=PLAN_TAG), plan))
    return handles


def recv_plan(ps: ParallelState) -> torch.Tensor:
    length = torch.empty(1, dtype=torch.int64)
    dist.recv(length, src=0, group=ps.control_group, tag=PLAN_TAG)
    plan = torch.empty(int(length.item()), dtype=torch.int64)
    dist.recv(plan, src=0, group=ps.control_group, tag=PLAN_TAG)
    return plan


def isend_result(ps: ParallelState, payload: torch.Tensor) -> tuple:
    """Sampling rank -> driver: one step's tokens (and debug top-k). Returns (work, tensor)."""
    return dist.isend(payload, dst=0, group=ps.control_group, tag=RESULT_TAG), payload


def irecv_result(ps: ParallelState, numel: int, src: int) -> tuple:
    """Posted by the driver when it submits the step, before the result can exist."""
    buf = torch.empty(numel, dtype=torch.int64)
    return dist.irecv(buf, src=src, group=ps.control_group, tag=RESULT_TAG), buf
