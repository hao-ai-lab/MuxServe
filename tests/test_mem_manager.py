import argparse
import torch
import time

from muxserve.flexstore.manager import FlexStoreManager
from muxserve.config import MuxServeConfig, JobConfig
from muxserve.zmq_utils import ZMQClient
from torch.multiprocessing.reductions import rebuild_cuda_tensor
from typing import Dict
from muxserve.logger import get_logger

logger = get_logger()

PORT = 10050


def test_reconstruction(queue):

    meta_data = queue.get()
    print("In child: Got meta_data from parent")

    print("### Begin to reconstruct model weight ...")
    rebuilt_weight = {}
    # print(meta_data)
    for rank, info in meta_data.items():
        for weight_name, weight_info in info.items():
            # print(f"#### Rebuilding {weight_name} in rank_{rank} ...")
            rebuilt_weight[f"{weight_name}_{rank}"] = rebuild_cuda_tensor(
                torch.Tensor, **weight_info)

    print(rebuilt_weight["model.embed_tokens.weight_0"])
    print(rebuilt_weight["model.embed_tokens.weight_1"])
    print(rebuilt_weight["model.embed_tokens.weight_0"].device)
    print(rebuilt_weight["model.embed_tokens.weight_1"].device)


def client_request_weight(pname: int):
    pname = f"child_{pname}"
    print(f"In {pname}: sleeping for a while ...")
    time.sleep(3)
    client = ZMQClient("localhost", PORT)
    for req in [
        ("weight", [0, "/mnt/afs/share/LLMCKPTs/huggyllama/llama-7b"]),
        ("weight", [1, "/mnt/afs/share/LLMCKPTs/huggyllama/llama-7b"]),
        ("weight", [0, "/mnt/afs/share/LLMCKPTs/huggyllama/llama-13b"]),
        ("weight", [1, "/mnt/afs/share/LLMCKPTs/huggyllama/llama-13b"]),
            "carpe_diem",
    ]:
        print(f"In {pname}: Sending {req} to memory manager")
        client.send_pyobj(req)
        data: Dict = client.recv_pyobj()
        print(f"In {pname}: Recieve metadata from memory manager")
        if type(data) == str and data == "Incorrect format":
            print(f"In {pname}: Bye ...")
            exit(0)
        for k, v in data.items():
            rebuilt = rebuild_cuda_tensor(torch.Tensor, **v)
            print(
                f"In {pname}: Rebuilding {req}_{k}: {rebuilt.shape}; {rebuilt.device}"
            )
        print(f"{'='*50}")


def client_request_kv_cache(pname: int):
    pname = f"child_{pname}"
    logger.info(f"In {pname}: sleeping for a while ...")
    # time.sleep(3)
    client = ZMQClient("localhost", PORT)

    blocktables = []
    for req in [
            # [dev_id, model_name]
        ("cache_init", [0, "/mnt/afs/share/LLMCKPTs/huggyllama/llama-7b"]),
        ("cache_init", [1, "/mnt/afs/share/LLMCKPTs/huggyllama/llama-7b"]),
        ("cache_init", [0, "/mnt/afs/share/LLMCKPTs/huggyllama/llama-13b"]),
        ("cache_init", [1, "/mnt/afs/share/LLMCKPTs/huggyllama/llama-13b"]),
            # [req_id, dev_id, model_name, num_blocks]
            # ("cache_alloc", [42, 0, "/mnt/afs/share/LLMCKPTs/huggyllama/llama-7b", 14]),
            # ("free_cache", [42, 0, "/mnt/afs/share/LLMCKPTs/huggyllama/llama-7b", [384, 352, 320, 288, 256, 224, 192, 160, 128, 96, 64, 32]]),
        ("cache_alloc",
         [42, 0, "/mnt/afs/share/LLMCKPTs/huggyllama/llama-7b", 12]),
        ("cache_alloc",
         [42, 0, "/mnt/afs/share/LLMCKPTs/huggyllama/llama-7b", 6]),
        ("cache_alloc",
         [42, 0, "/mnt/afs/share/LLMCKPTs/huggyllama/llama-13b", 1000]),
            # ("free_cache", [42, 0, "/mnt/afs/share/LLMCKPTs/huggyllama/llama-7b", [384, 352, 320]]),
            # ("free_cache", [42, 0, "/mnt/afs/share/LLMCKPTs/huggyllama/llama-13b", [920]]),
        ("cache_alloc",
         [42, 0, "/mnt/afs/share/LLMCKPTs/huggyllama/llama-13b", 1000]),
        ("cache_alloc",
         [42, 0, "/mnt/afs/share/LLMCKPTs/huggyllama/llama-13b", 700]),
        ("cache_alloc",
         [42, 0, "/mnt/afs/share/LLMCKPTs/huggyllama/llama-13b", 70]),
        ("cache_alloc",
         [42, 0, "/mnt/afs/share/LLMCKPTs/huggyllama/llama-13b", 1]),
        ("cache_alloc",
         [42, 0, "/mnt/afs/share/LLMCKPTs/huggyllama/llama-13b", 1]),
        ("cache_alloc",
         [42, 0, "/mnt/afs/share/LLMCKPTs/huggyllama/llama-13b", 1]),
            "carpe_diem",
    ]:
        logger.info(f"In {pname}: Sending {req} to memory manager")
        req_type = req[0]

        client.send_pyobj(req)

        data = client.recv_pyobj()
        logger.info(f"In {pname}: Recieve metadata from memory manager")

        if type(data) == str and data == "Incorrect format":
            print(f"In {pname}: Bye ...")
            exit(0)

        if req_type == "weight":
            for k, v in data.items():
                rebuilt = rebuild_cuda_tensor(torch.Tensor, **v)
                logger.info(
                    f"load_weight: In {pname}: "
                    f"Rebuilding {req[1][1]}_{k}: {rebuilt.shape}; {rebuilt.device}"
                )
        elif req_type == "cache_init":
            k_block, v_block = data
            rebuiltk = rebuild_cuda_tensor(torch.Tensor, **k_block)
            rebuiltv = rebuild_cuda_tensor(torch.Tensor, **v_block)
            logger.info(
                f"cache_init: rebuiltk: {rebuiltk.shape}; {rebuiltk.device}; {rebuiltk.data_ptr()}"
            )
            logger.info(
                f"cache_init: rebuiltv: {rebuiltv.shape}; {rebuiltv.device}; {rebuiltv.data_ptr()}"
            )
        elif req_type == "cache_alloc":
            logger.info(f"cache_alloc: cache_group indices: {data}")
            blocktables.append(data)
        elif req_type == "free_cache":
            logger.info(f"free_cache with: {req[1][-1]}")


def start_memory_manager(muxserve_config: MuxServeConfig):
    manager = FlexStoreManager(muxserve_config)
    manager.deploy()


def start_client(client_callback):
    child1 = mp.Process(target=client_callback, args=(1, ))
    child1.start()
    # child2 = mp.Process(target=client_request_weight, args=(2, ))
    # child2.start()


if __name__ == "__main__":

    import torch.multiprocessing as mp
    torch.multiprocessing.set_start_method('spawn')

    job_configs = [
        JobConfig(
            model="/mnt/afs/share/LLMCKPTs/huggyllama/llama-7b",
            pipeline_parallel_size=1,
            tensor_parallel_size=4,
            placement=[[0, 1, 2, 3]],
            mps_percentage=None,
        ),
        JobConfig(
            model="/mnt/afs/share/LLMCKPTs/huggyllama/llama-13b",
            pipeline_parallel_size=1,
            tensor_parallel_size=4,
            placement=[[0, 1, 2, 3]],
            mps_percentage=None,
        ),
    ]
    muxserve_config = MuxServeConfig(
        job_configs,
        num_gpus=2,
        mps_dir=None,
        block_size=16,
        gpu_memory_utilization=0.7,
        muxserve_host=None,
        flexstore_port=PORT,
        server_port=None,
        workload_config=None,
        max_num_batched_tokens=None,
        max_num_seqs=None,
        overload_threshold=None,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--server",

                        help="start mem server",

                        action="store_true")
    parser.add_argument("--client", help="start client", action="store_true")
    args = parser.parse_args()

    if args.server:
        start_memory_manager(muxserve_config)
    elif args.client:
        start_client(client_request_kv_cache)
