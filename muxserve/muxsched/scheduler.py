import os
import signal
import asyncio
import aiohttp
import enum
import json
import requests
import yaml
import time
import numpy as np
import subprocess
from multiprocessing import shared_memory, Pool
from typing import Any, Dict, Iterable, List, Tuple, Set

import torch
from transformers import AutoConfig
from muxserve.zmq_utils import ZMQClient
from muxserve.config import MuxServeConfig
from muxserve.constants import (SM_HOLD_NAME_FMT, ADD_REQ_NAME_FMT,
                              RET_REQ_NAME_FMT, PREEMPT_REQ_NAME_FMT)
from muxserve.flexserver.pipeworker import PipeWorker
from muxserve.muxsched.launcher import launch_flexserver_process
from muxserve.muxsched.resource import SMResource, SMPart
from muxserve.muxsched.workload_utils import get_workload, Workload, Request
from muxserve.shm_utils import (create_shared_var, read_shared_var,
                              write_shared_var, dump_to_shared_var,
                              load_from_shared_var, close_shared_var,
                              load_reqs_from_shared_var,
                              dump_reqs_to_shared_var)
from muxserve.tracer import FlexTracer, pack_to_proc_name
from muxserve.logger import get_logger

logger = get_logger()


class SchedStatus(enum.Enum):
    PREFILL_NOREQ = 0
    DECODE_NOSM = 1
    NO_MEM = 2
    NO_SM = 3
    RUNNING = 4
    NO_REQ = 5


def decode_response(response: Dict[str, Any]) -> str:
    output = response["text"][0]
    return output


async def async_post_http_request(prompt: str,
                                  request_id: str,
                                  api_url: str,
                                  is_free_cache: bool = False,
                                  max_tokens: int = 2048) -> Dict[str, Any]:
    headers = {"User-Agent": "Test Client"}

    pload = {
        "prompt": prompt,
        "request_id": request_id,
        "n": 1,
        "use_beam_search": False,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
        "is_free_cache": is_free_cache,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers,
                                json=pload) as response:
            data = await response.json()
    return data


class MuxScheduler:

    def __init__(self, muxserve_config: MuxServeConfig):
        self.muxserve_config = muxserve_config
        self.processes = {
            mps_percent: {}
            for mps_percent in [20, 30, 40, 50, 60, 70, 80, 90, 100]
        }
        self.num_gpus = self.muxserve_config.num_gpus
        self.pipeline_parallel_size = 1

        # resources
        self.sm_manager: SMResource = SMResource(
            overload_threshold=self.muxserve_config.overload_threshold)

        # workloads
        self._workload_queue: asyncio.PriorityQueue[Tuple[
            float, Request]] = asyncio.PriorityQueue()
        self.workload: Workload = None
        # we use a counter to track the number of requests status
        self.counter: Set[int] = set()
        self.is_finished = False

        # schedule queues
        self.lock = asyncio.Lock()
        self.running: Dict[str, List[Request]] = {}
        self.executing: Dict[str, Set[int]] = {}
        self.waiting: Dict[str, List[Request]] = {}
        self._preempted: Dict[str, List[Request]] = {}
        self.wait_for_cache: Dict[str, bool] = {}
        self.prefill_cannot_schedule: Dict[str, bool] = {}
        self.max_num_seqs: Dict[str, int] = {}
        self.hist_num_seqs: Dict[str, List[int]] = {}

        self.shm_size = 6
        self.sm_hold_name_to_shm = {}
        self.model_mps_dual = {}

        self._served_models = []
        self._name_to_model: Dict[str, str] = {}
        self._running_prefill_model_id = 0
        self._running_decoding_model_id = 0
        self._running_phase = 0

        # tracer
        self.enable_tracer = True

        # stats record
        self.sched_dict = {}
        self.prefill_batches: Dict[str, List[int]] = {}
        self.decoding_batches: Dict[str, List[int]] = {}

        self.enable_profiler = False
        self.start_batch = 50
        self.stop_batch = 150
        self.cur_batch = 0
        self.prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            with_stack=True,
            with_modules=True) if self.enable_profiler else None
        self.prof_out_name = f"log/profiler_muxserve/muxserve_schduler.json"

    def serve_models(self):
        """Serve all models with MPS processes."""
        np.random.seed(0)
        logger.info(f"MuxScheduler begins to serve models...")
        port = self.muxserve_config.server_port
        block_size = self.muxserve_config.block_size
        workload_file = self.muxserve_config.workload_config["workload_file"]
        ray_node_addr = self.muxserve_config.ray_node_address
        base_ray_port = self.muxserve_config.base_ray_port
        split_by_model = self.muxserve_config.workload_config.get(
            "split_by_model", None)
        if split_by_model is None:
            # only serve models in the model config
            split_by_model = []
            for model_id, job_config in enumerate(
                    self.muxserve_config.job_configs):
                model_name = job_config.name
                if model_name not in split_by_model:
                    split_by_model.append(model_name)
            split_by_model = ",".join(split_by_model)

        cluster_id = 0
        tracer_proc_names = []
        time.sleep(5)
        for model_id, job_config in enumerate(self.muxserve_config.job_configs):
            model_name = job_config.name
            self._name_to_model[model_name] = job_config.model
            self.pipeline_parallel_size = job_config.pipeline_parallel_size
            # self.max_num_seqs[model_name] = self.muxserve_config.max_num_seqs
            self.max_num_seqs[model_name] = job_config.max_num_seqs
            self.hist_num_seqs[model_name] = []
            assert len(job_config.mps_percentage) == 2
            for i, mps_percentage in enumerate(job_config.mps_percentage):
                name = SM_HOLD_NAME_FMT.format(model_name, mps_percentage)
                shm_var = create_shared_var(name,
                                            size=self.shm_size,
                                            create=True)
                self.sm_hold_name_to_shm[name] = shm_var

                dual_mps = job_config.mps_percentage[(i + 1) % 2]
                self.model_mps_dual[(model_name, mps_percentage)] = dual_mps

                is_prefill = mps_percentage == max(job_config.mps_percentage)
                self.processes[mps_percentage][model_name] = {}
                for dp_rank, placement in enumerate(job_config.placement):
                    ray_address = f"{ray_node_addr}:{base_ray_port + cluster_id}"
                    proc = launch_flexserver_process(
                        model_id,
                        model_name,
                        job_config.model,
                        self.muxserve_config.nnodes,
                        self.muxserve_config.nproc_per_node,
                        job_config.pipeline_parallel_size,
                        job_config.tensor_parallel_size,
                        block_size,
                        placement,
                        self.muxserve_config.flexstore_port,
                        self.muxserve_config.master_addr,
                        port,
                        mps_percentage,
                        self.muxserve_config.mps_dir,
                        workload_file,
                        split_by_model,
                        self.muxserve_config.max_num_batched_tokens,
                        # self.muxserve_config.max_num_seqs,
                        job_config.max_num_seqs,
                        is_prefill=is_prefill,
                        ray_address=ray_address,
                        schedule_approach=self.muxserve_config.schedule_approach)
                    cluster_id += 1
                self.processes[mps_percentage][model_name][dp_rank] = (proc,
                                                                       port)
                port += 1

                tracer_proc_names.append(
                    pack_to_proc_name(model_name, mps_percentage))

            self.running[model_name] = []
            self.executing[model_name] = set()
            self.waiting[model_name] = []
            self._preempted[model_name] = []
            self._served_models.append(model_name)
            self.wait_for_cache[model_name] = False
            self.prefill_cannot_schedule[model_name] = False
            self.prefill_batches[model_name] = []
            self.decoding_batches[model_name] = []
        self.decoding_batches["total"] = []
        self.prefill_batches["total"] = []
        self.tracer = FlexTracer(tracer_proc_names)

        # connect to flexstore
        self.tcp_client = ZMQClient("localhost",
                                    self.muxserve_config.flexstore_port)
        while True:
            self.tcp_client.send_pyobj(["init_finished", None])
            ret = self.tcp_client.recv_pyobj()
            if ret:
                break
            time.sleep(0.5)

        self.tcp_client.send_pyobj(["get_num_blocks", None])
        self.num_gpu_blocks = self.tcp_client.recv_pyobj()
        self.model_to_blocks_per_token: Dict[str, int] = {}

        for job_config in self.muxserve_config.job_configs:
            model_name = job_config.name
            model_config = AutoConfig.from_pretrained(job_config.model)
            tensor_parallel_size = job_config.tensor_parallel_size
            num_heads = model_config.num_attention_heads // tensor_parallel_size
            partition = PipeWorker.pipeline_split(
                model_config.num_hidden_layers,
                job_config.pipeline_parallel_size)
            num_hidden_layers = max(partition)
            self.model_to_blocks_per_token[
                model_name] = num_heads * num_hidden_layers

        logger.info(f"MuxScheduler finished serving models.")
        logger.info(f"Model Config:")
        print(f"{self.muxserve_config.model_config_path}:")
        print(f"{yaml.dump(self.muxserve_config.model_config)}")

        self.prepare_workloads()
        logger.info(f"MuxScheduler finished preparing workload.")

    def prepare_workloads(self):
        self.workload = self.get_workload()

        # get rate for each model
        self.model_rates = {}
        for (model_name, rate) in self.workload.workload_infos["rates"]:
            if model_name in self._served_models:
                self.model_rates[model_name] = rate

    def clean_up(self):
        logger.info(f"Clean processes...")
        # Clear shared variables.
        for job_config in self.muxserve_config.job_configs:
            model_name = job_config.name
            for mps_percentage in job_config.mps_percentage:
                for fmt in [
                        SM_HOLD_NAME_FMT, ADD_REQ_NAME_FMT, RET_REQ_NAME_FMT,
                        PREEMPT_REQ_NAME_FMT
                ]:
                    name = fmt.format(model_name, mps_percentage)
                    close_shared_var(name)

        # exit flexstore
        self.tcp_client.send_pyobj(["exit", None])

        # Terminate all MPS processes.
        for mps_percent in self.processes:
            for model_name in self.processes[mps_percent]:
                for dp_rank in self.processes[mps_percent][model_name]:
                    proc, _ = self.processes[mps_percent][model_name][dp_rank]

                    output = subprocess.check_output(
                        ['pgrep', '-P', str(proc.pid)])
                    child_pids = [int(pid) for pid in output.decode().split()]

                    # Terminate the child processes
                    for pid in child_pids:
                        os.kill(pid, signal.SIGTERM)
                    os.kill(proc.pid, signal.SIGTERM)
                    logger.info(f"Kill parent process {proc.pid}, "
                                f"child processes: {child_pids}")

    def get_workload(self) -> Workload:
        workload_file = self.muxserve_config.workload_config.get(
            "workload_file", None)
        if workload_file:
            workload = Workload.from_workload_file(workload_file)
        else:
            models = [
                job_config.name
                for job_config in self.muxserve_config.job_configs
            ]
            arrival_rates = self.muxserve_config.workload_config["arrival_rates"]
            start = self.muxserve_config.workload_config["start"]
            duration = self.muxserve_config.workload_config["duration"]
            dataset = self.muxserve_config.workload_config["dataset"]
            num_requests = self.muxserve_config.workload_config.get(
                "num_requests", None)
            workload = get_workload(models,
                                    arrival_rates,
                                    start,
                                    duration,
                                    dataset_path=dataset,
                                    num_requests=num_requests)

        split_by_model = self.muxserve_config.workload_config.get(
            "split_by_model", None)
        if split_by_model is not None:
            workload = workload.split_by_model(split_by_model)

        # only serve requests in the served models
        workload = workload.split_by_models(self._served_models)

        total_num_requests = 0
        for i in range(len(workload)):
            arrival_time = workload.arrivals[i]
            request = workload.requests[i]
            if request.model_name not in self._served_models:
                continue
            self._workload_queue.put_nowait((arrival_time, request))
            total_num_requests += 1
        # we use a counter to track the number of requests status
        self.counter = set([i for i in range(total_num_requests)])
        return workload

    def get_tick(self) -> float:
        return time.perf_counter() - self.cur_tick

    def adapt_max_num_seqs(self):
        models_utilization: List[str, float] = []
        models_under_utilized: List[str, int] = []
        steal_blocks_dict: Dict[str, int] = {}
        total_empty_blocks = 0

        blocks_per_models: Dict[str, int] = {}
        avg_blocks_per_seq_dict: Dict[str, int] = {}
        for model_name in self._served_models:
            max_bs = self.max_num_seqs[model_name]
            if len(self.hist_num_seqs[model_name]):
                avg_bs = np.mean(self.hist_num_seqs[model_name][-500:])
            else:
                avg_bs = 30
            utilization = avg_bs / max_bs

            avg_seq_len = self.avg_seq_len_dict[model_name]
            avg_blocks_per_seq = np.floor_divide(avg_seq_len,
                                                 self.muxserve_config.block_size)
            avg_blocks_per_seq = avg_blocks_per_seq * self.model_to_blocks_per_token[
                model_name]

            avg_blocks_per_seq_dict[model_name] = avg_blocks_per_seq
            blocks_per_models[model_name] = avg_blocks_per_seq * max_bs
            logger.info(
                f"Model {model_name} util {utilization} avg_blocks {avg_blocks_per_seq} max_bs {max_bs}"
            )
            if utilization >= 0.9:
                models_utilization.append((model_name, utilization))
                continue

            num_empty_blocks = int(max(
                1, (0.95 - utilization) * max_bs)) * avg_blocks_per_seq
            logger.info(
                f"    num_empty {num_empty_blocks} empty bs {(0.95 - utilization) * max_bs}"
            )

            models_under_utilized.append((model_name, num_empty_blocks))
            total_empty_blocks += num_empty_blocks
            steal_blocks_dict[model_name] = 0

        models_utilization.sort(key=lambda x: x[1], reverse=True)
        # steal from the utilized model
        for i in range(len(models_utilization)):
            model_name = models_utilization[i][0]

            num_waiting = len(self.waiting[model_name])
            num_running = len(self.running[model_name]) + len(
                self.executing[model_name])

            # avg_bs = np.mean(self.hist_num_seqs[model_name][-500:])

            if len(self.hist_num_seqs[model_name]):
                avg_bs = np.mean(self.hist_num_seqs[model_name][-500:])
            else:
                avg_bs = 30
            # not so many requests in the queue, skip
            if num_waiting + num_running < avg_bs * 1.1:
                continue
            tb_increased_bs = min(max(1, int(0.1 * avg_bs)),
                                  int(num_waiting + num_running - avg_bs))
            tb_increased_blocks = tb_increased_bs * avg_blocks_per_seq_dict[
                model_name]
            # if avg_bs * 0.1 < 1 and num_waiting + num_running > avg_bs + 1:
            #     tb_increased_blocks = max(
            #         tb_increased_blocks,
            #         int(total_empty_blocks * 0.9 / len(models_utilization)))

            incresad_blocks = 0
            for j in range(len(models_under_utilized)):
                src_model = models_under_utilized[j][0]
                steal_blocks = int(tb_increased_blocks *
                                   models_under_utilized[j][1] /
                                   total_empty_blocks)
                free_blocks = models_under_utilized[j][1] - steal_blocks_dict[
                    src_model]
                steal_blocks = min(steal_blocks, free_blocks)
                logger.info(
                    f"  model {model_name} steal {steal_blocks} from {src_model}, free_blocks {free_blocks}"
                )

                steal_blocks_dict[src_model] += steal_blocks
                incresad_blocks += steal_blocks
            blocks_per_models[model_name] += incresad_blocks

        # adjust blocks
        new_max_num_seqs = {}
        for i in range(len(models_under_utilized)):
            model_name = models_under_utilized[i][0]
            remain_blocks = blocks_per_models[model_name] - steal_blocks_dict[
                model_name]
            blocks_per_seq = avg_blocks_per_seq_dict[model_name]
            new_max_bs = np.floor_divide(remain_blocks, blocks_per_seq)
            # can we add one extra seq
            if remain_blocks % blocks_per_seq + self.num_fragment_blocks > blocks_per_seq:
                new_max_bs += 1
                self.num_fragment_blocks = self.num_fragment_blocks - (
                    blocks_per_seq - remain_blocks % blocks_per_seq)
            else:
                self.num_fragment_blocks += remain_blocks % blocks_per_seq
            new_max_num_seqs[model_name] = new_max_bs

        for i in range(len(models_utilization)):
            model_name = models_utilization[i][0]
            remain_blocks = blocks_per_models[model_name]
            blocks_per_seq = avg_blocks_per_seq_dict[model_name]
            new_max_bs = np.floor_divide(remain_blocks, blocks_per_seq)
            # can we add one extra seq
            if remain_blocks % blocks_per_seq + self.num_fragment_blocks > blocks_per_seq:
                new_max_bs += 1
                self.num_fragment_blocks = self.num_fragment_blocks - (
                    blocks_per_seq - remain_blocks % blocks_per_seq)
            else:
                self.num_fragment_blocks += remain_blocks % blocks_per_seq
            new_max_num_seqs[model_name] = new_max_bs

        # try to use fragment blocks
        models = sorted(self.waiting.keys(),
                        key=lambda x: len(self.waiting[x]),
                        reverse=True)
        for model_name in models:
            num_new_seqs = self.num_fragment_blocks // avg_blocks_per_seq_dict[
                model_name]
            new_max_num_seqs[model_name] += num_new_seqs
            self.num_fragment_blocks -= num_new_seqs * avg_blocks_per_seq_dict[
                model_name]

        # update max num seqs
        for model_name in self._served_models:
            logger.info(f"Adapt model {model_name} max num seqs from "
                        f"{self.max_num_seqs[model_name]} to "
                        f"{new_max_num_seqs[model_name]}")
            self.max_num_seqs[model_name] = new_max_num_seqs[model_name]

    def init_max_num_seqs(self, fixed=False) -> int:
        total_base = 0
        # sharegpt default
        avg_seq_len_dict = {
            model_name: 416
            for model_name in self._served_models
        }
        prompt_len = self.workload.workload_infos.get("prompt_len", None)
        output_len = self.workload.workload_infos.get("output_len", None)
        if prompt_len is not None and output_len is not None:
            if isinstance(prompt_len, int) or isinstance(prompt_len, float):
                prompt_len = [int(prompt_len)] * len(self._served_models)
            if isinstance(output_len, int) or isinstance(output_len, float):
                output_len = [int(output_len)] * len(self._served_models)
            for i in range(len(self._served_models)):
                model_name = self._served_models[i]
                if prompt_len[i] + output_len[i] > 0:
                    avg_seq_len_dict[
                        model_name] = prompt_len[i] + output_len[i]

        for (model_name, rate) in self.model_rates.items():
            group_size = self.model_to_blocks_per_token[model_name]
            total_base += group_size if fixed else rate * group_size
        for (model_name, rate) in self.model_rates.items():
            group_size = self.model_to_blocks_per_token[model_name]
            if fixed:
                ratio = group_size / total_base
            else:
                ratio = rate * group_size / total_base
            num_local_blocks = int(ratio * self.num_gpu_blocks)
            max_num_token_blocks = num_local_blocks // group_size

            avg_seq_len = avg_seq_len_dict[model_name]
            avg_blocks_per_seq = np.floor_divide(avg_seq_len,
                                                 self.muxserve_config.block_size)
            self.max_num_seqs[model_name] = np.floor_divide(
                max_num_token_blocks, avg_blocks_per_seq)
            logger.info(f"Model {model_name} Avg seq len {avg_seq_len} "
                        f"max num seqs {self.max_num_seqs[model_name]}")

        # record avg seq len
        self.avg_seq_len_dict = avg_seq_len_dict
        self.num_fragment_blocks = 0

    def switch_prefill_model(self, random: bool = False) -> str:
        if random:
            self._running_prefill_model_id = np.random.choice(
                list(range(len(self._served_models))))
        else:
            self._running_prefill_model_id = (self._running_prefill_model_id +
                                              1) % len(self._served_models)
        model = self._served_models[self._running_prefill_model_id]
        return model

    def switch_decoding_model(self, random: bool = False) -> str:
        if random:
            self._running_decoding_model_id = np.random.choice(
                list(range(len(self._served_models))))
        else:
            self._running_decoding_model_id = (
                self._running_decoding_model_id + 1) % len(self._served_models)
        model = self._served_models[self._running_decoding_model_id]
        return model

    def decrease_model(self, prefill=False) -> None:
        if prefill:
            self._running_prefill_model_id = (self._running_prefill_model_id -
                                              1) % len(self._served_models)
        else:
            self._running_decoding_model_id = (
                self._running_decoding_model_id - 1) % len(self._served_models)

    def switch_phase(self) -> str:
        self._running_phase = (self._running_phase + 1) % 2
        return "Prefill" if self._running_phase == 0 else "Decode"

    async def submit_workload(self, workload):
        # dispatch workload
        while True:
            if self.is_finished:
                break

            if self._workload_queue.empty():
                break
            (arrival_time, request) = self._workload_queue.get_nowait()
            if arrival_time > self.get_tick():
                self._workload_queue.put_nowait((arrival_time, request))
                await asyncio.sleep(0.003)
                continue
            async with self.lock:
                self.waiting[request.model_name].append(request)

    def estimate_mps(self, model_name, is_prefill):
        host = "127.0.0.1"
        # TODO: change to a smart schedule algorithm
        candidate_mps = []
        for mps_percentage, mps_processes in self.processes.items():
            if model_name in mps_processes:
                candidate_mps.append(mps_percentage)
        if len(candidate_mps) == 0:
            raise ValueError(f"Model {model_name} is not served.")

        if is_prefill:
            mps_percentage = max(candidate_mps)
        else:
            mps_percentage = min(candidate_mps)
        port = self.processes[mps_percentage][model_name][0][1]
        return mps_percentage // 10, host, port

    async def finish_requests(self):
        num_requests = len(self.workload)
        while True:
            if self.is_finished:
                break

            for (model_name, mps_percentage) in self.decode_names:
                name = PREEMPT_REQ_NAME_FMT.format(model_name, mps_percentage)
                req_ids = load_from_shared_var(name)
                async with self.lock:
                    for req_id in req_ids[::-1]:
                        req = self.workload.requests[req_id]
                        req.output_tokens = None
                        self.executing[req.model_name].remove(req_id)
                        self.waiting[req.model_name].insert(0, req)
                if req_ids:
                    self.wait_for_cache[model_name] = True
                    self.prefill_cannot_schedule[model_name] = True
                    logger.info(f"Preempte requests {req_ids} during decoding "
                                f"({model_name}), add them to waiting queue.")

            if len(self.counter) == 0:
                self.log_stats()
                self.is_finished = True
                break
            await asyncio.sleep(0.003)

    async def add_requests(self,
                           model_name: str,
                           mps_percentage: int,
                           requests: List[Request],
                           sm_res: List[SMPart],
                           is_prefill: bool = False):
        num_iters = 1

        # add requests
        shm_name = ADD_REQ_NAME_FMT.format(model_name, mps_percentage)
        tb_add_ids, tb_add_tokens = [], []
        async with self.lock:
            for req in requests:
                tb_add_ids.append(req.idx)
                if not is_prefill:
                    tb_add_tokens.append(req.output_tokens[-1])
                if req.submit_time is None:
                    req.submit_time = self.get_tick()
        if tb_add_ids:
            if is_prefill:
                dump_reqs_to_shared_var(shm_name, requests + tb_add_tokens)
            else:
                # if we've already sent the tokens(output_tokens > 1) to decoding proc before, we only need to sent the index of req to runtime
                dump_reqs_to_shared_var(shm_name, [
                    req.idx if len(req.output_tokens) > 1 else req
                    for req in requests
                ] + tb_add_tokens)

        # set sm_res
        name = SM_HOLD_NAME_FMT.format(model_name, mps_percentage)
        shm_var = self.sm_hold_name_to_shm[name]
        write_shared_var(shm_var, num_iters)
        self.holding_sm_mps.append((name, shm_var, sm_res))
        self.running_mps.add(name)
        if self.muxserve_config.schedule_approach == "fix":
            dual_mps = self.model_mps_dual[(model_name, mps_percentage)]
            dual_name = SM_HOLD_NAME_FMT.format(model_name, dual_mps)
            self.running_mps.add(dual_name)

        # trace events
        self.cur_batch += 1
        if self.enable_tracer:
            proc_name = pack_to_proc_name(model_name, mps_percentage)
            self.tracer.add_event("Forward",
                                  model_name,
                                  proc_name,
                                  self.get_tick(),
                                  start=True)

        if self.enable_profiler:
            if self.cur_batch == self.start_batch:
                self.prof.start()
            elif self.cur_batch == self.stop_batch:
                self.prof.stop()
                self.prof.export_chrome_trace(self.prof_out_name)
                logger.info(f"Export profiler output to {self.prof_out_name}")

    def can_schedule_prefill(self, model):
        waiting_queue = self.waiting[model]
        if len(waiting_queue) == 0:
            return False

        num_curr_seqs = len(self.executing[model]) + len(self.running[model])
        if num_curr_seqs >= self.max_num_seqs[model]:
            return False

        return True

    def schedule_prefill(self, model):
        waiting_queue = self.waiting[model]
        waiting_queue.sort()

        num_batched_token = 0
        num_curr_seqs = len(self.executing[model]) + len(self.running[model])
        scheduled = []
        while waiting_queue:
            request = waiting_queue[0]

            num_prompt_tokens = request.data[1]
            if (num_batched_token + num_prompt_tokens >
                    self.muxserve_config.max_num_batched_tokens):
                break

            if num_curr_seqs + 1 > self.max_num_seqs[model]:
                break

            request = waiting_queue.pop(0)
            num_batched_token += num_prompt_tokens
            num_curr_seqs += 1
            scheduled.append(request)
        return scheduled

    def schedule_decode(self, model):
        max_micro_num_seqs = self.max_num_seqs[
            model] // self.pipeline_parallel_size
        running_queue = self.running[model]
        running_queue.sort()

        running = []
        while running_queue:
            request = running_queue.pop(0)
            running.append(request)
            if len(running) >= max_micro_num_seqs:
                break
        return running

    async def try_schedule_prefill(self, model):
        num_sm, host, port = self.estimate_mps(model, True)

        sm_hold_name = SM_HOLD_NAME_FMT.format(model, num_sm * 10)
        if sm_hold_name in self.running_mps:
            status = SchedStatus.RUNNING
            return status

        if not self.sm_manager.can_allocate(num_sm, overload=True):
            status = SchedStatus.NO_SM
            return status

        # await asyncio.sleep(0)
        async with self.lock:
            if self.wait_for_cache[model]:
                status = SchedStatus.NO_MEM
                return status

            scheduled = self.schedule_prefill(model)
        if not scheduled:
            status = SchedStatus.PREFILL_NOREQ
            return status

        req_ids = [req.idx for req in scheduled]
        self.executing[model].update(req_ids)

        self.prefill_batches[model].append(len(scheduled))
        self.prefill_batches["total"].append(len(scheduled))
        logger.info(f"Schedule {model} prefill {len(scheduled)} requests: "
                    f"{req_ids}")
        sm_res = self.sm_manager.allocate(num_sm, overload=True)
        # launch tasks
        status = None
        await self.add_requests(model, num_sm * 10, scheduled, sm_res, True)
        return status

    async def try_schedule_decoding(self, model):
        num_sm, host, port = self.estimate_mps(model, False)

        sm_hold_name = SM_HOLD_NAME_FMT.format(model, num_sm * 10)
        if sm_hold_name in self.running_mps:
            status = SchedStatus.RUNNING
            return status

        if not self.sm_manager.can_allocate(num_sm, overload=True):
            status = SchedStatus.NO_SM
            return status

        async with self.lock:
            running = self.schedule_decode(model)
            if not running:
                status = SchedStatus.NO_REQ
                return status
            req_ids = [req.idx for req in running]
            self.executing[model].update(req_ids)

        self.decoding_batches[model].append(len(running))
        self.decoding_batches["total"].append(len(running))
        logger.info(f"Schedule {model} decoding {len(running)} requests: "
                    f"{req_ids}")
        sm_res = self.sm_manager.allocate(num_sm, overload=True)
        # launch tasks
        status = None
        await self.add_requests(model, num_sm * 10, running, sm_res, False)
        return status

    async def fcfs_schedule(self, prefill_in_exec: bool,
                            has_prefill_to_schedule: bool,
                            last_sched_time: float, last_warn_time: float):
        # decoding schedule order
        schedule_order = []
        for i in range(len(self._served_models)):
            model = self._served_models[i]
            if len(self.running[model]) == 0:
                continue

            self.running[model].sort()
            tstamp = self.running[model][0].idx
            schedule_order.append((tstamp, model))
        schedule_order.sort()
        can_schedule_prefill = True
        if len(schedule_order) > 0:
            model = schedule_order[0][1]
            can_schedule_prefill = not self.prefill_cannot_schedule[model]

        status = None
        if not prefill_in_exec and can_schedule_prefill:
            tb_scheduled_model = None
            earilest_submit_time = float("inf")
            for i in range(len(self._served_models)):
                model = self._served_models[i]
                if len(self.waiting[model]) == 0:
                    continue

                self.waiting[model].sort()
                if self.waiting[model][0].idx < earilest_submit_time:
                    earilest_submit_time = self.waiting[model][0].idx
                    tb_scheduled_model = model

            # try to schedule prefill
            if tb_scheduled_model is not None:
                status = await self.try_schedule_prefill(tb_scheduled_model)
                if status is None:
                    last_sched_time = self.get_tick()
                    prefill_in_exec = True
                else:
                    self.warn_log(last_sched_time, last_warn_time, status,
                                  model, True)
                    # wait for resources
                    if status == SchedStatus.NO_SM:
                        return status, prefill_in_exec, last_sched_time, last_warn_time

        # try to schedule decoding
        for (tstamp, model) in schedule_order:
            status = await self.try_schedule_decoding(model)
            if status is None:
                last_sched_time = self.get_tick()
            else:
                if self.warn_log(last_sched_time, last_warn_time, status,
                                 model, False):
                    last_warn_time = self.get_tick()
                    logger.info(
                        f"SM status: {self.sm_manager.num_free_sms} free "
                        f"{self.sm_manager.num_overloaded_sms} overloaded")
                    logger.info(f"Unfinished requests: {sorted(self.counter)}")
                if status == SchedStatus.NO_SM:
                    return status, prefill_in_exec, last_sched_time, last_warn_time
        return status, prefill_in_exec, last_sched_time, last_warn_time

    async def roundrobin_schedule(self, prefill_in_exec: bool,
                                  has_prefill_to_schedule: bool,
                                  last_sched_time: float,
                                  last_warn_time: float):
        has_prefill_to_schedule = False
        scheduled_model_id = self._running_prefill_model_id
        for i in range(len(self._served_models)):
            model = self.switch_prefill_model()
            status = await self.try_schedule_prefill(model)
            if status is None:
                last_sched_time = self.get_tick()
                scheduled_model_id = self._running_prefill_model_id
                prefill_in_exec = True
            else:
                if status == SchedStatus.NO_SM and self.can_schedule_prefill(
                        model) and not self.wait_for_cache[model]:
                    scheduled_model_id = self._running_prefill_model_id - 1
                    has_prefill_to_schedule = True
                    break
                self.warn_log(last_sched_time, last_warn_time, status, model,
                              True)
        self._running_prefill_model_id = scheduled_model_id

        if not prefill_in_exec and not has_prefill_to_schedule:
            for i in range(len(self._served_models)):
                model = self._served_models[i]
                if self.wait_for_cache[model]:
                    continue
                if self.can_schedule_prefill(model):
                    has_prefill_to_schedule = True
                    break

        if has_prefill_to_schedule and not prefill_in_exec:
            await asyncio.sleep(0 if status is None else 0.0001)
            return prefill_in_exec, has_prefill_to_schedule, last_sched_time, last_warn_time

        scheduled_model_id = self._running_decoding_model_id
        for i in range(len(self._served_models)):
            model = self.switch_decoding_model()
            status = await self.try_schedule_decoding(model)
            if status is None:
                last_sched_time = self.get_tick()
                scheduled_model_id = self._running_decoding_model_id
            else:
                if self.warn_log(last_sched_time, last_warn_time, status,
                                 model, False):
                    last_warn_time = self.get_tick()
                    logger.info(
                        f"SM status: {self.sm_manager.num_free_sms} free "
                        f"{self.sm_manager.num_overloaded_sms} overloaded")
                    logger.info(f"Unfinished requests: {sorted(self.counter)}")
        self._running_decoding_model_id = scheduled_model_id

        return prefill_in_exec, has_prefill_to_schedule, last_sched_time, last_warn_time

    async def adbs_schedule(self, prefill_in_exec: bool,
                            has_prefill_to_schedule: bool,
                            last_sched_time: float, last_warn_time: float):
        has_prefill_to_schedule = False
        scheduled_model_id = self._running_prefill_model_id
        for i in range(len(self._served_models)):
            model = self.switch_prefill_model()
            status = await self.try_schedule_prefill(model)
            if status is None:
                last_sched_time = self.get_tick()
                scheduled_model_id = self._running_prefill_model_id
                prefill_in_exec = True
            else:
                self.warn_log(last_sched_time, last_warn_time, status, model,
                              True)
        self._running_prefill_model_id = scheduled_model_id

        if not prefill_in_exec and not has_prefill_to_schedule:
            for i in range(len(self._served_models)):
                model = self._served_models[i]
                if self.wait_for_cache[model]:
                    continue
                if self.can_schedule_prefill(model):
                    has_prefill_to_schedule = True
                    break

        if has_prefill_to_schedule and not prefill_in_exec:
            await asyncio.sleep(0 if status is None else 0.0001)
            return prefill_in_exec, has_prefill_to_schedule, last_sched_time, last_warn_time

        scheduled_model_id = self._running_decoding_model_id

        for i in range(len(self._served_models)):
            model = self.switch_decoding_model()
            status = await self.try_schedule_decoding(model)
            if status is None:
                last_sched_time = self.get_tick()
                scheduled_model_id = self._running_decoding_model_id
            else:
                if self.warn_log(last_sched_time, last_warn_time, status,
                                 model, False):
                    last_warn_time = self.get_tick()
                    logger.info(
                        f"SM status: {self.sm_manager.num_free_sms} free "
                        f"{self.sm_manager.num_overloaded_sms} overloaded")
                    logger.info(f"Unfinished requests: {sorted(self.counter)}")
        self._running_decoding_model_id = scheduled_model_id

        return prefill_in_exec, has_prefill_to_schedule, last_sched_time, last_warn_time

    def warn_log(self, last_sched_tick, last_warn_tick, status, model,
                 prefill):
        if self.get_tick() - last_sched_tick > 35 and self.get_tick(
        ) - last_warn_tick > 35:
            tag = "prefill" if prefill else "decode"
            logger.info(f"Fail to schedule {tag} requests due to "
                        f"{status} for {model}: "
                        f"waiting {len(self.waiting[model])} "
                        f"running {len(self.running[model])} "
                        f"executing {len(self.executing[model])} ")
            return True
        return False

    async def schedule_requests(self):
        self.holding_sm_mps = []
        self.running_mps = set()
        num_requests = len(self.workload)

        # setup scheduler
        need_adapt = False
        if self.muxserve_config.schedule_approach == "fcfs":
            sched_func = self.fcfs_schedule
        elif self.muxserve_config.schedule_approach == "roundrobin":
            sched_func = self.roundrobin_schedule
        elif self.muxserve_config.schedule_approach == "adbs":
            self.init_max_num_seqs()
            sched_func = self.adbs_schedule
            need_adapt = True
            adapt_interval = 8
        elif self.muxserve_config.schedule_approach == "fix-adbs":
            self.init_max_num_seqs(fixed=True)
            sched_func = self.adbs_schedule
        elif self.muxserve_config.schedule_approach == "fix":
            self.init_max_num_seqs(fixed=True)
            sched_func = self.adbs_schedule
        else:
            assert False, f"Unknown schedule approach {self.muxserve_config.schedule_approach}"

        last_adapt_time = self.get_tick()
        last_sched_time = last_warn_time = self.get_tick()
        prefill_in_exec, has_prefill_to_schedule = False, False
        while True:
            if self.is_finished:
                break

            # release SMs
            holding_sm_mps = self.holding_sm_mps
            self.holding_sm_mps = []
            cur_time = self.get_tick()
            for (shm_name, shm_var, sm_res) in holding_sm_mps:
                ret = read_shared_var(shm_var)
                if ret <= 0:
                    self.sm_manager.free(sm_res)
                    self.running_mps.remove(shm_name)
                    if self.muxserve_config.schedule_approach == "fix":
                        model_name, _, mps_percentage = shm_name.split("_")
                        dual_mps = self.model_mps_dual[(model_name,
                                                        int(mps_percentage))]
                        dual_name = SM_HOLD_NAME_FMT.format(
                            model_name, dual_mps)
                        self.running_mps.remove(dual_name)

                    # trace events
                    if self.enable_tracer:
                        avg_bs = abs(ret)
                        # hack
                        model_name, _, mps_percentage = shm_name.split("_")
                        proc_name = pack_to_proc_name(model_name,
                                                      mps_percentage)
                        self.tracer.add_event(f"BS {avg_bs}",
                                              model_name,
                                              proc_name,
                                              cur_time,
                                              start=False)
                else:
                    self.holding_sm_mps.append((shm_name, shm_var, sm_res))

            # deal with NO Cache for prefill
            for (model_name, mps_percentage) in self.prefill_names:
                name = PREEMPT_REQ_NAME_FMT.format(model_name, mps_percentage)
                req_ids = load_from_shared_var(name)
                if req_ids:
                    async with self.lock:
                        for req_id in req_ids[::-1]:
                            req = self.workload.requests[req_id]
                            self.executing[req.model_name].remove(req_id)
                            self.waiting[req.model_name].insert(0, req)
                        self.wait_for_cache[model_name] = True
                    prefill_in_exec = False

                    logger.info(f"Preempte requests {req_ids} during prefill "
                                f"({model_name}), add them to waiting queue.")

            for (model_name, mps_percentage) in self.prefill_names:
                name = RET_REQ_NAME_FMT.format(model_name, mps_percentage)
                batch_reqs = load_from_shared_var(name)
                if not batch_reqs:
                    continue
                async with self.lock:
                    num_reqs = len(batch_reqs) // 2
                    for i in range(num_reqs):
                        req_id = batch_reqs[i]
                        output_token = batch_reqs[i + num_reqs]

                        req = self.workload.requests[req_id]
                        req.prefill_end_time = self.get_tick()
                        req.output_tokens = [output_token]
                        self.executing[req.model_name].remove(req_id)

                        if len(req.output_tokens) == req.data[2]:
                            req.end_time = self.get_tick()
                            self.counter.remove(req.idx)

                            finished = num_requests - len(self.counter)
                            if finished % 10 == 0:
                                logger.info(
                                    f"Finish {finished}/{num_requests} requests"
                                )
                        else:
                            self.running[req.model_name].append(req)
                prefill_in_exec = False

            for (model_name, mps_percentage) in self.decode_names:
                name = RET_REQ_NAME_FMT.format(model_name, mps_percentage)
                batch_reqs = load_from_shared_var(name)
                if not batch_reqs:
                    continue
                has_req_finished = False
                async with self.lock:
                    num_reqs = len(batch_reqs) // 2
                    self.hist_num_seqs[model_name].append(num_reqs)
                    for i in range(num_reqs - 1, -1, -1):
                        req_id = batch_reqs[i]
                        output_token = batch_reqs[i + num_reqs]

                        req = self.workload.requests[req_id]
                        self.executing[req.model_name].remove(req_id)
                        req.output_tokens.append(output_token)
                        if len(req.output_tokens) == req.data[2]:
                            req.end_time = self.get_tick()
                            self.counter.remove(req.idx)
                            has_req_finished = True

                            finished = num_requests - len(self.counter)
                            if finished % 10 == 0:
                                logger.info(
                                    f"Finish {finished}/{num_requests} requests"
                                )
                        else:
                            self.running[req.model_name].insert(0, req)
                if has_req_finished:
                    self.prefill_cannot_schedule[model_name] = False
                    for key in self.wait_for_cache:
                        self.wait_for_cache[key] = False

            ret = await sched_func(prefill_in_exec, has_prefill_to_schedule,
                                   last_sched_time, last_warn_time)
            status, prefill_in_exec, last_sched_time, last_warn_time = ret

            if self.get_tick() - last_sched_time > 60 * 4:
                logger.info("Scheduler Timeout Error, Exit!")
                self.is_finished = True
                break

            if need_adapt and self.get_tick(
            ) - last_adapt_time > adapt_interval:
                self.adapt_max_num_seqs()
                last_adapt_time = self.get_tick()
                adapt_interval = 8

            await asyncio.sleep(0 if status is None else 0.0001)

    async def schedule_loop(self):
        # sleep a while for proc warmup
        time.sleep(20)

        prefill_names, decode_names = [], []
        for job_config in self.muxserve_config.job_configs:
            model_name = job_config.name
            for mps_percentage in job_config.mps_percentage:
                if mps_percentage == max(job_config.mps_percentage):
                    prefill_names.append((model_name, mps_percentage))
                else:
                    decode_names.append((model_name, mps_percentage))
        self.prefill_names = prefill_names
        self.decode_names = decode_names

        logger.info(f"MuxScheduler Begin to schedule requests, "
                    f"total {len(self.workload)} requests.")
        self.cur_tick = time.perf_counter()
        await asyncio.gather(
            self.submit_workload(self.workload),
            self.schedule_requests(),
            self.finish_requests(),
        )
        logger.info(f"Finish all requests begin to exit...")
        self.clean_up()

    def log_stats(self):
        logger.info(f"Finish all requests, total {len(self.workload)}")

        for job_config in self.muxserve_config.job_configs:
            model_name = job_config.name
            self.sched_dict[model_name] = {}
            self.sched_dict[model_name]["model_name"] = job_config.model
            self.sched_dict[model_name]["mps"] = job_config.mps_percentage
            self.sched_dict[model_name]["first_token_latency"] = 0
            self.sched_dict[model_name]["output_per_token_latency"] = 0
            self.sched_dict[model_name]["request_num"] = 0
            # self.sched_dict[model_name]["all_latency"] = []

        self.sched_dict["first_token_latency"] = 0
        self.sched_dict["output_per_token_latency"] = 0
        self.sched_dict["request_num"] = 0

        model_to_requests = {}
        for i in range(len(self.workload)):
            req = self.workload.requests[i]
            if req.model_name not in model_to_requests:
                model_to_requests[req.model_name] = []
            model_to_requests[req.model_name].append(req)

            sched_lat = req.submit_time - self.workload.arrivals[i]
            prefill_lat = req.prefill_end_time - req.submit_time
            decode_lat = req.end_time - req.prefill_end_time
            logger.info(
                f"Request {req.idx} model {req.model_name} "
                f"prompt {req.data[1]} output {req.data[2]} "
                f"arrival {self.workload.arrivals[i]:.3f} "
                f"submit {req.submit_time:.3f} "
                f"prefill_end {req.prefill_end_time:.3f} "
                f"end {req.end_time:.3f} "
                f"sched_lat {sched_lat:.3f} prefill_lat {prefill_lat:.3f} "
                f"decode_lat {decode_lat:.3f} ")

        logger.info("")
        logger.info("Workload Statistics:")
        if self.workload.workload_infos:
            rates = self.workload.workload_infos.pop("rates")
            for (model, rate) in rates:
                logger.info(f"  Model: {model} rate: {rate}")
            for key, value in self.workload.workload_infos.items():
                logger.info(f"{key}: {value}")
        logger.info("")

        total_time, total_token = 0, 0
        avg_lat, first_token_lat, avg_per_output_token_lat = 0, 0, 0
        latency_list, ttft_list, tpot_list = [], [], []
        for model_name, requests in model_to_requests.items():
            total_num_tokens = sum(
                [req.data[1] + req.data[2] for req in requests])
            elapsed_time = max([req.end_time for req in requests])
            req_tpt = len(requests) / elapsed_time
            token_tpt = total_num_tokens / elapsed_time
            total_time = max(total_time, elapsed_time)
            total_token += total_num_tokens

            latency_list_per_model = [
                (req.end_time - self.workload.arrivals[req.idx])
                for req in requests
            ]
            p99 = np.percentile(latency_list_per_model, 99)
            p95 = np.percentile(latency_list_per_model, 95)
            p90 = np.percentile(latency_list_per_model, 90)
            latency_list.extend(latency_list_per_model)

            weight = len(requests) / len(self.workload)
            avg_lat_per_model = np.mean(latency_list_per_model)

            tpot_list_per_model = [
                (req.end_time - req.prefill_end_time) / req.data[2]
                for req in requests
            ]
            avg_per_output_token_lat_per_model = np.mean(tpot_list_per_model)
            p99_tpot = np.percentile(tpot_list_per_model, 99)
            p95_tpot = np.percentile(tpot_list_per_model, 95)
            p90_tpot = np.percentile(tpot_list_per_model, 90)
            tpot_list.extend(tpot_list_per_model)

            ttft_list_per_model = [
                (req.prefill_end_time - self.workload.arrivals[req.idx])
                for req in requests
            ]
            first_token_lat_per_model = np.mean(ttft_list_per_model)
            p99_ttft = np.percentile(ttft_list_per_model, 99)
            p95_ttft = np.percentile(ttft_list_per_model, 95)
            p90_ttft = np.percentile(ttft_list_per_model, 90)
            ttft_list.extend(ttft_list_per_model)

            avg_lat += avg_lat_per_model * weight
            avg_per_output_token_lat += avg_per_output_token_lat_per_model * weight
            first_token_lat += first_token_lat_per_model * weight

            logger.info(
                f"Name: {model_name} \n"
                f"Model: {self._name_to_model[model_name]} \n"
                f"Throughput {req_tpt:.2f} requests/s {token_tpt:.2f} tokens/s \n"
                f"avg req latency: {avg_lat_per_model:.3f} \n"
                f"avg latency of first token: {first_token_lat_per_model:.3f} \n"
                f"avg latency per output token: {avg_per_output_token_lat_per_model:.3f} \n"
                f"[avg latency] p99: {p99:.3f}, p95: {p95:.3f}, p90: {p90:.3f} \n"
                f"[TTFT] p99: {p99_ttft:.3f}, p95: {p95_ttft:.3f}, p90: {p90_ttft:.3f} \n"
                f"[TPOT] p99: {p99_tpot:.3f}, p95: {p95_tpot:.3f}, p90: {p90_tpot:.3f} \n"
                f"Prefill avg batches: {np.mean(self.prefill_batches[model_name]):.3f}, all times: {len(self.prefill_batches[model_name])} \n"
                f"Decoding avg batches: {np.mean(self.decoding_batches[model_name]):.3f}, all times: {len(self.decoding_batches[model_name])} \n"
            )

            self.sched_dict[model_name]["throughput"] = req_tpt
            self.sched_dict[model_name]["tokens_throughput"] = token_tpt

            self.sched_dict[model_name][
                "first_token_latency"] = first_token_lat_per_model
            self.sched_dict[model_name][
                "output_per_token_latency"] = avg_per_output_token_lat_per_model

            self.sched_dict[model_name]["prefill_avg_batches"] = np.mean(
                self.prefill_batches[model_name])
            self.sched_dict[model_name]["prefill_batch_times"] = len(
                self.prefill_batches[model_name])
            self.sched_dict[model_name]["decoding_avg_batches"] = np.mean(
                self.decoding_batches[model_name])
            self.sched_dict[model_name]["decoding_batch_times"] = len(
                self.decoding_batches[model_name])

            self.sched_dict[model_name]["p99[avg_latency]"] = p99
            self.sched_dict[model_name]["p95[avg_latency]"] = p95
            self.sched_dict[model_name]["p90[avg_latency]"] = p90

            self.sched_dict[model_name]["p99[TTFT]"] = p99_ttft
            self.sched_dict[model_name]["p95[TTFT]"] = p95_ttft
            self.sched_dict[model_name]["p90[TTFT]"] = p90_ttft

            self.sched_dict[model_name]["p99[TPOT]"] = p99_tpot
            self.sched_dict[model_name]["p95[TPOT]"] = p95_tpot
            self.sched_dict[model_name]["p90[TPOT]"] = p90_tpot

            self.sched_dict[model_name]["request_num"] = len(requests)
            self.sched_dict["request_num"] += len(requests)

        p99 = np.percentile(latency_list, 99)
        p95 = np.percentile(latency_list, 95)
        p90 = np.percentile(latency_list, 90)
        p99_ttft = np.percentile(ttft_list, 99)
        p95_ttft = np.percentile(ttft_list, 95)
        p90_ttft = np.percentile(ttft_list, 90)
        p99_tpot = np.percentile(tpot_list, 99)
        p95_tpot = np.percentile(tpot_list, 95)
        p90_tpot = np.percentile(tpot_list, 90)
        req_tpt = len(self.workload) / total_time
        token_tpt = total_token / total_time
        logger.info(
            f"Summary: Throughput {req_tpt:.2f} "
            f"requests/s {token_tpt:.2f} tokens/s \n"
            f"avg req latency: {avg_lat:.3f} \n"
            f"avg latency of first token: {first_token_lat:.3f} \n"
            f"avg latency per output token: {avg_per_output_token_lat:.3f} \n"
            f"[avg latency] p99: {p99:.3f}, p95: {p95:.3f}, p90: {p90:.3f} \n"
            f"[TTFT] p99: {p99_ttft:.3f}, p95: {p95_ttft:.3f}, p90: {p90_ttft:.3f} \n"
            f"[TPOT] p99: {p99_tpot:.3f}, p95: {p95_tpot:.3f}, p90: {p90_tpot:.3f} \n"
            f"Prefill avg batches: {np.mean(self.prefill_batches['total']):.3f}, all times: {len(self.prefill_batches['total'])} \n"
            f"Decoding avg batches: {np.mean(self.decoding_batches['total']):.3f}, all times: {len(self.decoding_batches['total'])} \n"
        )

        self.sched_dict["throughput"] = req_tpt
        self.sched_dict["tokens_throughput"] = token_tpt
        self.sched_dict["first_token_latency"] = first_token_lat
        self.sched_dict["output_per_token_latency"] = avg_per_output_token_lat

        self.sched_dict["prefill_avg_batches"] = np.mean(
            self.prefill_batches["total"])
        self.sched_dict["prefill_batch_times"] = len(
            self.prefill_batches["total"])
        self.sched_dict["decoding_avg_batches"] = np.mean(
            self.decoding_batches["total"])
        self.sched_dict["decoding_batch_times"] = len(
            self.decoding_batches["total"])

        self.sched_dict["p99[avg_latency]"] = p99
        self.sched_dict["p95[avg_latency]"] = p95
        self.sched_dict["p90[avg_latency]"] = p90

        self.sched_dict["p99[TTFT]"] = p99_ttft
        self.sched_dict["p95[TTFT]"] = p95_ttft
        self.sched_dict["p90[TTFT]"] = p90_ttft

        self.sched_dict["p99[TPOT]"] = p99_tpot
        self.sched_dict["p95[TPOT]"] = p95_tpot
        self.sched_dict["p90[TPOT]"] = p90_tpot

        # Compute the latency statistics.
        for prefix in ["(w/ sched)", "(w/o sched)"]:
            print()
            add_sched = (prefix == "(w/ sched)")
            for model_name, requests in model_to_requests.items():
                logger.info(f"Model: {model_name}")
                fn = lambda x: self.workload.arrivals[
                    x.idx] if add_sched else x.submit_time
                request_latency = [
                    (req.data[1], req.data[2], req.end_time - fn(req),
                     req.end_time - req.prefill_end_time) for req in requests
                ]

                avg_latency = np.mean(
                    [latency for _, _, latency, _ in request_latency])
                logger.info(f"{prefix} Average latency: {avg_latency:.2f} s")

                avg_per_token_latency = np.mean([
                    latency / (prompt_len + output_len)
                    for prompt_len, output_len, latency, _ in request_latency
                ])
                logger.info(f"{prefix} Average latency per token: "
                            f"{avg_per_token_latency:.3f} s")

                avg_per_output_token_latency = np.mean([
                    latency / output_len
                    for _, output_len, _, latency in request_latency
                ])
                logger.info(f"{prefix} Average latency per output token: "
                            f"{avg_per_output_token_latency:.3f} s")

        if self.enable_tracer:
            self.tracer.export("log/muxserve_trace.json")

        workload_file = self.muxserve_config.workload_config.get(
            "workload_file", None)

        # if workload_file:
        #     PREFIX = os.environ.get("FLEXSM_SHM_PREFIX", "")
        #     with open(workload_file[:-5] + f"_{PREFIX}_stats.json", "w") as f:
        #         json.dump(self.sched_dict, f, indent=2)
