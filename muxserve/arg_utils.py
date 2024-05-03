import argparse
import dataclasses
import yaml
import torch
from torch.distributed.argparse_util import env
from dataclasses import dataclass
from typing import Optional

from muxserve.config import JobConfig, MuxServeConfig

DTYPE_MAP = {"fp16": torch.float16}


@dataclass
class MuxServeArgs:
    """Arguments for muxserve."""
    model_config: str
    mps_dir: Optional[str] = None
    ray_node_address: str = "127.0.0.1"
    base_ray_port: int = 6379
    num_ray_cluster: int = 4
    block_size: int = 32
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    muxserve_host: str = "127.0.0.1"
    flexstore_port: int = 50051
    server_port: int = 50060
    workload_file: str = None
    split_by_model: str = None
    schedule_approach: bool = "adbs"
    nnodes: str = "1"
    nproc_per_node: str = "1"
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--model-config',
                            type=str,
                            help='path of the serving job config file')
        parser.add_argument('--mps-dir',
                            type=str,
                            default=MuxServeArgs.mps_dir,
                            help='path of the mps directory')
        parser.add_argument('--ray-node-address',
                            type=str,
                            default=MuxServeArgs.ray_node_address,
                            help='ray node address')
        parser.add_argument('--base-ray-port',
                            type=int,
                            default=MuxServeArgs.base_ray_port,
                            help='the base port of ray cluster')
        parser.add_argument('--num-ray-cluster',
                            type=int,
                            default=MuxServeArgs.num_ray_cluster,
                            help='the number of ray cluster')
        # FlexStore arguments.
        parser.add_argument('--block-size',
                            type=int,
                            default=MuxServeArgs.block_size,
                            choices=[8, 16, 32],
                            help='token block size')
        parser.add_argument('--gpu-memory-utilization',
                            type=float,
                            default=MuxServeArgs.gpu_memory_utilization,
                            help='the percentage of GPU memory to be used for'
                            'the flexstore')
        parser.add_argument('--max-num-batched-tokens',
                            type=int,
                            default=MuxServeArgs.max_num_batched_tokens,
                            help='maximum number of batched tokens per '
                            'iteration')
        parser.add_argument('--max-num-seqs',
                            type=int,
                            default=MuxServeArgs.max_num_seqs,
                            help='maximum number of sequences per iteration')
        parser.add_argument('--muxserve-host',
                            type=str,
                            default=MuxServeArgs.muxserve_host,
                            help='the host address of flexstore')
        parser.add_argument('--flexstore-port',
                            type=int,
                            default=MuxServeArgs.flexstore_port,
                            help='the port of flexstore')
        # MuxScheduler arguments.
        parser.add_argument('--server-port',
                            type=int,
                            default=MuxServeArgs.server_port,
                            help='the port of vllm server')
        parser.add_argument('--workload-file',
                            type=str,
                            default=MuxServeArgs.workload_file,
                            help='the path of workload file')
        parser.add_argument('--split-by-model',
                            type=str,
                            default=MuxServeArgs.split_by_model,
                            help='split the workload by model')
        parser.add_argument(
            '--schedule-approach',
            type=str,
            default=MuxServeArgs.schedule_approach,
            choices=["roundrobin", "adbs", "fcfs", "fix", "fix-adbs"],
            help='schedule approach')
        # launch configs
        parser.add_argument(
            "--nnodes",
            type=str,
            default=MuxServeArgs.nnodes,
            help="Number of nodes",
        )
        parser.add_argument(
            "--nproc-per-node",
            "--nproc_per_node",
            type=str,
            default=MuxServeArgs.nproc_per_node,
            help=
            "Number of workers per node; supported values: [auto, cpu, gpu, int].",
        )
        parser.add_argument(
            "--node-rank",
            "--node_rank",
            type=int,
            action=env,
            default=MuxServeArgs.node_rank,
            help="Rank of the node for multi-node distributed training.",
        )
        parser.add_argument(
            "--master-addr",
            "--master_addr",
            default=MuxServeArgs.master_addr,
            type=str,
            action=env,
            help=
            "Address of the master node (rank 0) that only used for static rendezvous. It should "
            "be either the IP address or the hostname of rank 0. For single node multi-proc training "
            "the --master-addr can simply be 127.0.0.1; IPv6 should have the pattern "
            "`[0:0:0:0:0:0:0:1]`.",
        )
        parser.add_argument(
            "--master-port",
            "--master_port",
            default=MuxServeArgs.master_port,
            type=int,
            action=env,
            help=
            "Port on the master node (rank 0) to be used for communication during distributed "
            "training. It is only used for static rendezvous.",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'MuxServeArgs':
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        muxserve_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return muxserve_args

    def create_muxserve_config(self) -> MuxServeConfig:
        assert self.model_config is not None, "model_config is not specified"
        with open(self.model_config, "r") as f:
            model_config = yaml.safe_load(f)
        # overwrite max_num_seqs
        self.max_num_seqs = model_config["max_num_seqs"]
        self.overload_threshold = model_config["overload_threshold"]
        self.gpu_memory_utilization = model_config["gpu_memory_utilization"]
        num_gpus = model_config["num_gpus"]

        if self.split_by_model is not None:
            print(f"{'='*30} Split By Model ({self.split_by_model}) {'='*30}")
            model_config["workloads"]["split_by_model"] = self.split_by_model
        else:
            assert model_config["workloads"].get("split_by_model", None) is \
                None, "split_by_model shouldn't be specified in config"

        job_configs = []
        for model in model_config["models"]:
            if self.split_by_model is not None and \
                    model["name"] != self.split_by_model:
                continue
            job_cfg = JobConfig(
                name=model["name"],
                model=model["model"],
                pipeline_parallel_size=model["pipeline_parallel_size"],
                tensor_parallel_size=model["tensor_parallel_size"],
                placement=model["placement"],
                mps_percentage=model["mps_percentage"],
                max_num_seqs=getattr(model, "max_num_seqs",
                                     model_config["max_num_seqs"]),
                model_dtype=DTYPE_MAP[model["model_dtype"]],
            )
            if self.split_by_model is not None:
                num_gpus = len(model["placement"][0])
            job_configs.append(job_cfg)
            assert model["pipeline_parallel_size"] * model[
                "tensor_parallel_size"] <= num_gpus, f"Exceeds {num_gpus} GPUs"
        assert len(job_configs) > 0, "No job is specified"

        if self.workload_file is not None:
            assert model_config["workloads"].get(
                "workload_file") is None, "workload_file is specified twice"
            model_config["workloads"]["workload_file"] = self.workload_file
        else:
            assert model_config["workloads"].get(
                "workload_file") is not None, "workload_file is not specified"
            self.workload_file = model_config["workloads"]["workload_file"]

        muxserve_config = MuxServeConfig(
            job_configs=job_configs,
            num_gpus=num_gpus,
            ray_node_address=self.ray_node_address,
            base_ray_port=self.base_ray_port,
            num_ray_cluster=self.num_ray_cluster,
            mps_dir=self.mps_dir,
            block_size=self.block_size,
            overload_threshold=self.overload_threshold,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            muxserve_host=self.muxserve_host,
            flexstore_port=self.flexstore_port,
            server_port=self.server_port,
            workload_config=model_config["workloads"],
            model_config=model_config,
            model_config_path=self.model_config,
            schedule_approach=self.schedule_approach,
            nnodes=int(self.nnodes),
            nproc_per_node=int(self.nproc_per_node),
            node_rank=self.node_rank,
            master_addr=self.master_addr,
            master_port=self.master_port)
        return muxserve_config
