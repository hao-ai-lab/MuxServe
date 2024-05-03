import os
import argparse
import copy
import subprocess
import time
from typing import List

from vllm.zmq_tool import ZMQClient

from muxserve.arg_utils import MuxServeArgs
from muxserve.config import MuxServeConfig
from muxserve.muxsched.launcher import launch_flexserver_process


def launch_flexstore(model_config: str, rest_args: List[str]):
    proc_env = copy.deepcopy(os.environ)

    cmd = [
        "python", "-m", "muxserve.entrypoint", "--flexstore", "--model-config",
        f"{model_config}"
    ] + rest_args
    print(f"Launch flexstore: {' '.join(cmd)}")
    flexstore_proc = subprocess.Popen(
        cmd,
        env=proc_env,
    )
    return flexstore_proc


def launch_flexserver(muxserve_config: MuxServeConfig, logdir: str):
    port = muxserve_config.server_port
    block_size = muxserve_config.block_size
    workload_file = muxserve_config.workload_config["workload_file"]
    split_by_model = muxserve_config.workload_config.get("split_by_model", None)
    flexstore_port = muxserve_config.flexstore_port

    # Although we use for loop here, we only launch one flexserver process
    for model_id, job_config in enumerate(muxserve_config.job_configs):
        model_name = job_config.name
        for mps_percentage in job_config.mps_percentage:
            is_prefill = True
            for dp_rank, placement in enumerate(job_config.placement):
                model = job_config.model.split("/")[-1]
                logfile = f"{logdir}/{model}_n{len(placement)}_mps{mps_percentage}.log"
                flexserver_proc = launch_flexserver_process(
                    model_id,
                    model_name,
                    job_config.model,
                    muxserve_config.nnodes,
                    muxserve_config.nproc_per_node,
                    job_config.pipeline_parallel_size,
                    job_config.tensor_parallel_size,
                    block_size,
                    placement,
                    flexstore_port,
                    muxserve_config.master_addr,
                    port,
                    mps_percentage,
                    muxserve_config.mps_dir,
                    workload_file,
                    split_by_model,
                    muxserve_config.max_num_batched_tokens,
                    muxserve_config.max_num_seqs,
                    is_prefill=is_prefill,
                    runtime_profile=True,
                    logfile=logfile,
                )
                yield flexserver_proc
                port += 1


def build_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "model_config",
        type=str,
        help="path of the serving job config file",
    )
    return parser


def main():
    parser = argparse.ArgumentParser(description='MuxServe launcher')
    parser = build_parser(parser)
    parser.add_argument("rest",
                        nargs=argparse.REMAINDER,
                        help="Arguments for the script")

    args = parser.parse_args()

    muxserve_parser = argparse.ArgumentParser(description='MuxServe launcher')
    muxserve_parser = build_parser(muxserve_parser)
    muxserve_parser = MuxServeArgs.add_cli_args(muxserve_parser)
    muxserve_parser.add_argument(
        "--log-dir",
        type=str,
        help="path to save log files",
    )
    muxserve_args = muxserve_parser.parse_args([args.model_config] + args.rest)
    log_dir = muxserve_args.log_dir

    rest_args = []
    last = -1
    for i in range(len(args.rest)):
        if args.rest[i] == "--log-dir" or i == last:
            last = i + 1
            continue
        rest_args.append(args.rest[i])
    flexstore_proc = launch_flexstore(args.model_config, rest_args)

    muxserve_args = MuxServeArgs.from_cli_args(muxserve_args)
    muxserve_config = muxserve_args.create_muxserve_config()
    flexstore_port = muxserve_config.flexstore_port
    tcp_client = ZMQClient('localhost', flexstore_port)
    num_ready_processes = 0
    for flexserver_proc in launch_flexserver(muxserve_config, log_dir):
        while True:
            tcp_client.send_pyobj(["query_num_ready_processes", None])
            ret = tcp_client.recv_pyobj()
            if ret == num_ready_processes + 1:
                num_ready_processes += 1
                break
            time.sleep(0.5)

        tcp_client.send_pyobj(["init_finished", None])
        ret = tcp_client.recv_pyobj()
        if ret:
            break

        flexserver_proc.terminate()
        time.sleep(5)

    flexserver_proc.terminate()
    flexstore_proc.terminate()

    # wait for user cancel command, and check the status of the process periodically
    while True:
        try:
            flexstore_proc.wait(1)
            flexserver_proc.wait(1)
        except subprocess.TimeoutExpired:
            continue
        else:
            break


if __name__ == '__main__':
    main()
