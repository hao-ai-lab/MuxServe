import os
import argparse
import copy
import subprocess
import time
from typing import List

from torch.distributed.argparse_util import env


class MuxServeLauncher:
    """Launch FlexStore and MuxScheduler process for muxserve."""

    def __init__(self, nnodes: int, nproc_per_node: int, node_rank: int,
                 master_addr: str, master_port: int, model_config: str,
                 rest_args: List[str]):
        self.nnodes = nnodes
        self.nproc_per_node = nproc_per_node
        self.node_rank = node_rank
        self.master_addr = master_addr
        self.master_port = master_port
        self.model_config = model_config
        self.rest_args = rest_args
        self.flexstore_proc = None
        self.muxsched_proc = None

    def launch_flexstore(self):
        """Launch FlexStore process."""
        proc_env = copy.deepcopy(os.environ)

        # each node we will launch one flexstore process
        self.flexstore_proc = subprocess.Popen(
            [
                "python", "-m", "muxserve.entrypoint", "--nnodes",
                f"{self.nnodes}", "--nproc_per_node", f"{self.nproc_per_node}",
                "--node_rank", f"{self.node_rank}", "--master_addr",
                f"{self.master_addr}", "--master_port", f"{self.master_port}",
                "--flexstore", "--model-config", f"{self.model_config}"
            ] + self.rest_args,
            env=proc_env,
        )

    def launch_muxsched(self):
        """Launch MuxScheduler process."""
        proc_env = copy.deepcopy(os.environ)

        self.muxsched_proc = subprocess.Popen(
            [
                "python", "-m", "muxserve.entrypoint", "--nnodes",
                f"{self.nnodes}", "--nproc_per_node", f"{self.nproc_per_node}",
                "--node_rank", f"{self.node_rank}", "--master_addr",
                f"{self.master_addr}", "--master_port", f"{self.master_port}",
                "--muxscheduler", "--model-config", f"{self.model_config}"
            ] + self.rest_args,
            env=proc_env,
        )

    def terminate_flexstore(self):
        """Terminate FlexStore process."""
        self.flexstore_proc.terminate()

    def terminate_muxsched(self):
        """Terminate MuxScheduler process."""
        self.muxsched_proc.terminate()

    def __exit__(self, exc_type, exc_value, traceback):
        """Terminate FlexStore and MuxScheduler process."""
        self.terminate_flexstore()
        self.terminate_muxsched()


def muxserve_serve():
    parser = argparse.ArgumentParser(description='MuxServe launcher')
    parser.add_argument(
        "--nnodes",
        type=str,
        default="1",
        help="Number of nodes",
    )
    parser.add_argument(
        "--nproc-per-node",
        "--nproc_per_node",
        type=str,
        default="1",
        help=
        "Number of workers per node; supported values: [auto, cpu, gpu, int].",
    )
    parser.add_argument(
        "--node-rank",
        "--node_rank",
        type=int,
        action=env,
        default=0,
        help="Rank of the node for multi-node distributed training.",
    )
    parser.add_argument(
        "--master-addr",
        "--master_addr",
        default="127.0.0.1",
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
        default=29500,
        type=int,
        action=env,
        help=
        "Port on the master node (rank 0) to be used for communication during distributed "
        "training. It is only used for static rendezvous.",
    )
    parser.add_argument(
        "model_config",
        type=str,
        help="path of the serving job config file",
    )
    parser.add_argument("rest",
                        nargs=argparse.REMAINDER,
                        help="Arguments for the script")

    args = parser.parse_args()

    muxserve_launcher = MuxServeLauncher(args.nnodes, args.nproc_per_node,
                                     args.node_rank, args.master_addr,
                                     args.master_port, args.model_config,
                                     args.rest)
    muxserve_launcher.launch_flexstore()
    muxserve_launcher.launch_muxsched()

    # wait for user cancel command, and check the status of the process periodically
    while True:
        try:
            muxserve_launcher.flexstore_proc.wait(1)
            muxserve_launcher.muxsched_proc.wait(1)
        except subprocess.TimeoutExpired:
            continue
        else:
            break


if __name__ == '__main__':
    muxserve_serve()
