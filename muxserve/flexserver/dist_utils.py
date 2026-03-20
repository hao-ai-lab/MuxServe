import os
import time
from typing import Optional

import torch

from vllm.config import ParallelConfig
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel)
from vllm.model_executor import set_random_seed

from muxserve.logger import get_logger

logger = get_logger()


def _setup_env():
    use_openmpi = os.environ.get("OMPI_COMM_WORLD_SIZE", None) is not None
    use_mpich = os.environ.get("PMI_SIZE", None) is not None
    if use_openmpi:
        local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
        rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
        world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
    elif use_mpich:
        local_rank = int(os.environ.get('MPI_LOCALRANKID', 0))
        rank = int(os.environ.get('PMI_RANK', 0))
        world_size = int(os.environ.get('PMI_SIZE', 1))

    if use_mpich or use_openmpi:
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)


def _init_distributed_environment(
    parallel_config: ParallelConfig,
    seed: int,
    distributed_init_method: Optional[str] = None,
) -> None:
    """Initialize the distributed environment."""
    assert (parallel_config.tensor_parallel_size == 1
            or parallel_config.pipeline_parallel_size
            == 1), "Support one parallelism only!"
    rank = int(os.getenv("RANK", "-1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size}).")
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    logger.info(f"Rank {rank} initialize distributed environment with "
                f"TP={parallel_config.tensor_parallel_size}, "
                f"PP={parallel_config.pipeline_parallel_size}, "
                f"LOCAL_RANK={local_rank}, "
                f"WORLD_SIZE={parallel_config.world_size}, "
                f"MASTER_ADDR={os.environ['MASTER_ADDR']}, "
                f"MASTER_PORT={os.environ['MASTER_PORT']}")
    initialize_model_parallel(parallel_config.tensor_parallel_size,
                              parallel_config.pipeline_parallel_size)

    logger.info(f"Rank {rank} set seed to {seed}")
    set_random_seed(seed)


def all_reduce_latency(nbytes, group=None):
    buf = torch.randn(nbytes // 2, dtype=torch.float16).cuda()
    torch.cuda.synchronize()
    # warmup
    for _ in range(5):
        torch.distributed.all_reduce(buf, group=group)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(25):
        torch.distributed.all_reduce(buf, group=group)
    torch.cuda.synchronize()
    end = time.perf_counter()
    avg_speed = (end - begin) * 1e6 / 25

    iter_speeds = []
    for _ in range(25):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        torch.distributed.all_reduce(buf, group=group)
        torch.cuda.synchronize()
        end = time.perf_counter()
        iter_speeds.append((end - begin) * 1e6)

    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)
    if rank == 0:
        algobw = nbytes / avg_speed / 1e3
        busbw = algobw * 2 * (world_size - 1) / world_size
        logger.info(f"{nbytes:15d}({nbytes / 1024 / 1024:8.2f}MB): "
                    f"{avg_speed:8.3f}us {min(iter_speeds):8.3f}us "
                    f"{algobw:4.2f}GB/s {busbw:4.2f}GB/s")


def test_all_reduce():
    tensor_sizes = [2**i for i in range(2, 29)]
    for payload in tensor_sizes:
        all_reduce_latency(payload)
