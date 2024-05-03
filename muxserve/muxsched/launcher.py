import os
import copy
import subprocess
from muxserve.logger import get_logger

logger = get_logger()


def launch_flexserver_process(model_id,
                              name,
                              model,
                              nnodes,
                              nproc_per_node,
                              pipeline_parallel_size,
                              tensor_parallel_size,
                              block_size,
                              placement,
                              flexstore_port,
                              master_addr,
                              master_port,
                              mps_percentage,
                              mps_dir,
                              workload_file,
                              split_by_model,
                              max_num_batched_tokens,
                              max_num_seqs,
                              is_prefill=False,
                              ray_address=None,
                              runtime_profile=False,
                              logfile=None,
                              schedule_approach=None):
    prefill_option = "--is-prefill" if is_prefill else ""
    split_option = f"--split-by-model {split_by_model}" if split_by_model else ""
    runtime_profile_option = "--runtime-profile" if runtime_profile else ""

    cmd = f"python -m torch.distributed.launch " \
          f"--nnodes={nnodes} " \
          f"--nproc-per-node={nproc_per_node} " \
          f"--master-addr {master_addr} " \
          f"--master-port {master_port} " \
          f"muxserve/flexserver/muxserve_server.py " \
          f"--model-id {model_id} --model-name {name} " \
          f"--model {model} --tensor-parallel-size {tensor_parallel_size} " \
          f"--pipeline-parallel-size {pipeline_parallel_size} " \
          f"--block-size {block_size} --swap-space 1 " \
          f"--max-num-batched-tokens {max_num_batched_tokens} " \
          f"--max-num-seqs {max_num_seqs} " \
          f"--flexstore-port {flexstore_port} --workload-file {workload_file} " \
          f"--mps-percentage {mps_percentage} {prefill_option} {split_option} " \
          f"{runtime_profile_option} "

    proc_env = copy.deepcopy(os.environ)
    # proc_env["CUDA_VISIBLE_DEVICES"] = ",".join([str(g) for g in placement])
    if mps_dir is not None:
        proc_env["CUDA_MPS_PIPE_DIRECTORY"] = f"{mps_dir}/nvidia-mps"
        proc_env["CUDA_MPS_LOG_DIRECTORY"] = f"{mps_dir}/nvidia-log"
        if schedule_approach == "fix":
            real_mps = mps_percentage - 10 if is_prefill else mps_percentage
        else:
            real_mps = mps_percentage
        proc_env[f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(real_mps)
    proc_env["MASTER_ADDR"] = master_addr
    proc_env["MASTER_PORT"] = str(master_port)

    logdir = os.environ.get("VLLM_PROC_LOG", "log/vllm_proc")
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    model_name = name.split("/")[-1]
    if logfile is None:
        logfile = f"{logdir}/{model_name}_sm{mps_percentage}.log"
    logger.info(f"Start process cmd: {cmd}, Output log file: {logfile}")

    logfile_writer = open(logfile, "w")
    logfile_writer.write(f"Start process cmd: {cmd}\n")
    logfile_writer.write(f"Environment Variable: \n")
    for k, v in proc_env.items():
        logfile_writer.write(f"    {k}: {v}\n")
    proc = subprocess.Popen(
        cmd,
        env=proc_env,
        shell=True,
        stdout=logfile_writer,
        stderr=subprocess.STDOUT,
    )
    return proc
