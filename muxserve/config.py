import torch
from typing import List, Dict, Any


class JobConfig:
    """Configuration for one job.

    Args:
        model: Name or path of the huggingface model to use.
        pipeline_parallel_size: Number of pipeline parallel groups.
        tensor_parallel_size: Number of tensor parallel groups.
    """

    def __init__(self,
                 name: str,
                 model: str,
                 pipeline_parallel_size: int,
                 tensor_parallel_size: int,
                 placement: List[List[int]],
                 mps_percentage: List[int],
                 max_num_seqs: int,
                 model_dtype: torch.dtype = torch.float16):
        self.name = name
        self.model = model
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.placement = placement
        self.mps_percentage = mps_percentage
        self.max_num_seqs = max_num_seqs
        self.model_dtype = model_dtype


class MuxServeConfig:
    """Configuration for muxserve.

    Args:
        job_configs: List of JobConfig.
        num_gpus: Number of GPUs to use.
        block_size: Token block size.
        gpu_memory_utilization: The percentage of GPU memory to be used for the
            flexstore.
    """

    def __init__(self, job_configs: List[JobConfig], num_gpus: int,
                 ray_node_address: str, base_ray_port: int,
                 num_ray_cluster: int, mps_dir: str, block_size: int,
                 overload_threshold: int, gpu_memory_utilization: float,
                 max_num_batched_tokens: int, max_num_seqs: int,
                 muxserve_host: str, flexstore_port: int, server_port: int,
                 workload_config: Dict[str, Any], model_config: Dict[Any, Any],
                 model_config_path: str, schedule_approach: bool, nnodes: int,
                 nproc_per_node: int, node_rank: int, master_addr: str,
                 master_port: int):
        self.job_configs = job_configs
        self.num_gpus = num_gpus
        self.ray_node_address = ray_node_address
        self.base_ray_port = base_ray_port
        self.num_ray_cluster = num_ray_cluster
        self.mps_dir = mps_dir
        self.block_size = block_size
        self.overload_threshold = overload_threshold
        self.gpu_memory_utilization = gpu_memory_utilization
        if max_num_batched_tokens is not None:
            self.max_num_batched_tokens = max_num_batched_tokens
        else:
            self.max_num_batched_tokens = 2048
        self.max_num_seqs = max_num_seqs
        self.muxserve_host = muxserve_host
        self.flexstore_port = flexstore_port
        self.server_port = server_port
        self.workload_config = workload_config
        self.model_config = model_config
        self.model_config_path = model_config_path
        self.schedule_approach = schedule_approach
        self.nnodes = nnodes
        self.nproc_per_node = nproc_per_node
        self.node_rank = node_rank
        self.master_addr = master_addr
        self.master_port = master_port

        self.head_size = 128

        self.num_runtime_processes = 0
        for job_config in self.job_configs:
            self.num_runtime_processes += len(job_config.mps_percentage)
