"""Non-driver ranks, and a launcher for running every rank on one machine.

A worker loads its shard of the model, sizes and allocates its KV cache in agreement with every other
rank (block ids are global, so all ranks must hold the same number of blocks), then executes the
driver's step plans until told to stop. Under torchrun every rank runs `python -m engine.api.server`
and all but rank 0 end up here. For tests and offline runs, `local_cluster` makes the calling process
rank 0 and starts the other ranks as child processes.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import multiprocessing as mp
import os
import signal
import socket

import torch

from engine.config import EngineConfig
from engine.core.model_runner import ModelRunner
from engine.core.sampler import Sampler
from engine.distributed.executor import worker_loop
from engine.distributed.parallel import ParallelState, init_parallel, shutdown_parallel
from engine.model.loader import load_config, load_model, resolve_model_path

logger = logging.getLogger(__name__)


def run_worker(config: EngineConfig, parallel: ParallelState) -> None:
    # A worker's lifetime is the driver's: the driver stops it with a shutdown plan once it has drained
    # the batches in flight. A Ctrl-C reaches every rank, and torchrun forwards SIGINT and SIGTERM to
    # every rank; a worker that died of it would leave the driver waiting on it. If the driver itself
    # dies, torchrun kills the rest.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    path = resolve_model_path(config.model)
    config.resolve(load_config(path).max_position_embeddings)
    model = load_model(path, config.device, config.torch_dtype, parallel)
    sampler = Sampler(config.device, config.seed)  # used only on the sampling rank
    runner = ModelRunner(model, config, parallel)
    runner.size_and_allocate(sampler)
    logger.info("rank %d (stage %d, tp rank %d) ready", parallel.rank, parallel.pp_rank, parallel.tp_rank)
    worker_loop(parallel, runner, sampler, config.block_size)


def device_type(config: EngineConfig) -> str:
    if config.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if config.device.startswith("cuda") else "cpu"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawned_worker(config_fields: dict, rank: int, port: int, threads: int) -> None:
    config = EngineConfig(**config_fields)
    torch.set_num_threads(threads)
    parallel = init_parallel(
        config.tensor_parallel_size, config.pipeline_parallel_size, device_type(config),
        rank=rank, world_size=config.world_size, init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        run_worker(config, parallel)
    finally:
        shutdown_parallel()


@contextlib.contextmanager
def local_cluster(config: EngineConfig):
    """Make this process rank 0 and start ranks 1..world-1 as local child processes. Yields the
    ParallelState to build an LLMEngine with; the caller must call engine.shutdown() before leaving
    the block. Everything binds to 127.0.0.1."""
    port = _free_port()
    threads = max(1, (os.cpu_count() or 2) // (2 * config.world_size))
    ctx = mp.get_context("spawn")
    fields = {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
    procs = [ctx.Process(target=_spawned_worker, args=(fields, rank, port, threads), daemon=True)
             for rank in range(1, config.world_size)]
    for p in procs:
        p.start()
    parallel = init_parallel(
        config.tensor_parallel_size, config.pipeline_parallel_size, device_type(config),
        rank=0, world_size=config.world_size, init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        yield parallel
    finally:
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                p.kill()
        shutdown_parallel()
        failed = [p.exitcode for p in procs if p.exitcode not in (0, None)]
        if failed:
            raise RuntimeError(f"worker processes exited with codes {failed}")
