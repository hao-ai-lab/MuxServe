import os
import time
import numpy as np

from typing import List, Tuple, Union, Optional, Any, Dict

import torch
import torch.distributed as dist

from muxserve.batch_scheduler import BatchScheduler
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.model_executor import InputMetadata
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import (detokenize_incrementally,
                                               get_tokenizer)
from vllm.core.scheduler import Scheduler
from vllm.engine.llm_engine import LLMEngine
from vllm.model_executor.parallel_utils import parallel_state
from vllm.utils import Counter
from vllm.zmq_tool import ZMQClient

from muxserve.constants import PREEMPT_REQ_NAME_FMT
from muxserve.shm_utils import (dump_to_shared_var, load_from_shared_var,
                              write_list_to_shared_var,
                              read_list_from_shared_var, close_shared_var)
from muxserve.flexserver.p2p_communication import (send_forward, recv_forward,
                                                 PipelineParallelConfig)
from muxserve.logger import get_logger

logger = get_logger()

KVCache = Tuple[torch.Tensor, torch.Tensor]

def log_gpu_memory_usage(show_log=False, empty_cache=False, message=""):
    def gpu_memory_usage_logger(func):
        def wrapper(*args, **kwargs):
            if show_log:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
                peak_memory = (total_gpu_memory - free_gpu_memory) / (1024**3)
                prev_reserved = torch.cuda.memory_reserved() / (1024**3)
                prev_max_reserved = torch.cuda.max_memory_reserved() / (1024**3)
                prev_allocated = torch.cuda.memory_allocated() / (1024**3)
                prev_max_allocated = torch.cuda.max_memory_allocated() / (1024**3)

                # log_message = (f"[BEFORE RUN] reserved: {prev_reserved:.3f} GB, max_reserved: {prev_max_reserved:.3f} GB, "
                #                f"allocated: {prev_allocated:.3f} GB, max_allocated: {prev_max_allocated:.3f} GB, "
                #                f"Peak memory: {peak_memory:.3f} GB")
                # logger.info(log_message)

            result = func(*args, **kwargs)

            if empty_cache:
                torch.cuda.empty_cache()

            if show_log:
                # torch.cuda.synchronize()
                free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
                peak_memory = (total_gpu_memory - free_gpu_memory) / (1024**3) - peak_memory
                prev_reserved = torch.cuda.memory_reserved() / (1024**3) - prev_reserved
                prev_max_reserved = torch.cuda.max_memory_reserved() / (1024**3) - prev_max_reserved
                prev_allocated = torch.cuda.memory_allocated() / (1024**3) - prev_allocated
                prev_max_allocated = torch.cuda.max_memory_allocated() / (1024**3) - prev_max_allocated

                #     _, batch_reqs, _ = result
                # BS = len(batch_reqs)

                log_message = (f"{message}"
                               f"[AFTER RUN] allocated: {prev_allocated:.3f} GB, max_allocated: {prev_max_allocated:.3f} GB, "
                               f"reserved: {prev_reserved:.3f} GB, max_reserved: {prev_max_reserved:.3f} GB, "
                               f"Peak memory: {peak_memory:.3f} GB")
                logger.info(log_message)

            return result

        return wrapper

    return gpu_memory_usage_logger

class LLMRuntime(LLMEngine):

    def __init__(
        self,
        model_id: int,
        model_name: str,
        tp_rank: int,
        stage_id: int,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        log_stats: bool = True,
        tcp_client: Optional[ZMQClient] = None,
        runtime_profile: bool = False,
    ) -> None:
        logger.info(
            "Initializing an LLM runtime with config: "
            f"model_id={model_id}, "
            f"model_name={model_name!r}, "
            f"model={model_config.model!r}, "
            f"tokenizer={model_config.tokenizer!r}, "
            f"tokenizer_mode={model_config.tokenizer_mode}, "
            f"revision={model_config.revision}, "
            f"tokenizer_revision={model_config.tokenizer_revision}, "
            f"trust_remote_code={model_config.trust_remote_code}, "
            f"dtype={model_config.dtype}, "
            f"max_seq_len={model_config.max_model_len}, "
            f"download_dir={model_config.download_dir!r}, "
            f"load_format={model_config.load_format}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"quantization={model_config.quantization}, "
            f"seed={model_config.seed})")

        self.model_id = model_id
        self.model_name = model_name
        self.model_config = model_config
        self.cache_config = cache_config
        assert self.cache_config.sliding_window == getattr(
            self.model_config.hf_config, "sliding_window", None)
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.log_stats = log_stats

        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
            tokenizer_revision=model_config.tokenizer_revision,
            revision=model_config.revision)

        self.tp_rank = tp_rank
        self.stage_id = stage_id
        self.num_stages = self.parallel_config.pipeline_parallel_size
        self.tensor_parallel_size = self.parallel_config.tensor_parallel_size
        self.world_size = self.parallel_config.world_size
        self.pipeline_config = PipelineParallelConfig(
            variable_seq_lengths=True)
        self.global_steps = 0

        mps = self.model_config.mps_percentage
        self.preempt_req_shm_name = PREEMPT_REQ_NAME_FMT.format(
            self.model_name, mps)
        if self.tp_rank == 0:
            self.tp_share_name = [
                f"m{self.model_id}_mps{mps}_r{i + 1}"
                for i in range(self.tensor_parallel_size - 1)
            ]
        else:
            self.tp_share_name = f"m{self.model_id}_mps{mps}_r{self.tp_rank}"
        self.send_shm_name = f"m{self.model_id}_mps{mps}_pp_s{self.stage_id}"
        prev_stage_id = self.stage_id - 1
        self.recv_shm_name = f"m{self.model_id}_mps{mps}_pp_s{prev_stage_id}"
        self.block_size = self.cache_config.block_size

        self._init_workers(stage_id, model_id)
        self.num_layers = self.workers[0].num_hidden_layers
        self.num_heads = self.model_config.get_num_kv_heads(
            self.parallel_config)

        # Profile the memory usage and initialize the cache.
        self._init_cache()

        # tcp client for muxserve
        self.tcp_client = tcp_client
        self.is_muxserve = (self.tcp_client is not None)

        parallel_config = self.parallel_config
        max_seq_len = self.model_config.max_model_len
        if max_seq_len is None:
            max_seq_len = 2048
        self.batch_scheduler = BatchScheduler(
            self.model_config.get_num_layers(parallel_config),
            self.model_config.get_num_kv_heads(parallel_config), max_seq_len,
            self.cache_config.block_size)

        # Logging.
        self.last_logging_time = 0.0
        # List of (timestamp, num_tokens)
        self.num_prompt_tokens: List[Tuple[float, int]] = []
        # List of (timestamp, num_tokens)
        self.num_generation_tokens: List[Tuple[float, int]] = []

        # warmup engine
        if self.tcp_client is not None and self.stage_id == 0 and self.tp_rank == 0:
            while True:
                self.tcp_client.send_pyobj(["start_warmup", None])
                status = self.tcp_client.recv_pyobj()
                if status:
                    break
                time.sleep(3)

        logger.info("Warmup engine...")
        self.unique_id = Counter(int(os.environ["MASTER_PORT"]) * 100)
        for _ in range(5):
            self.warmup()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Do some profling
        if runtime_profile:
            self.profile()
            # pytorch profiler to generate profiling trace
            # self.torch_profiler(8, 128)

        if self.tcp_client is not None and self.stage_id == 0 and self.tp_rank == 0:
            self.tcp_client.send_pyobj(["warmup_ready", None])
            self.tcp_client.recv_pyobj()
        logger.info("LLM Engine begins serving...")

    def get_allocated_cache_blocks(self):
        while self.scheduler.waiting:
            seq_group = self.scheduler.waiting.pop(0)
            # loop until we find a pre-allocated cache block
            # walkaround for race condition
            show_sleep = True
            while True:
                ret_status = self.scheduler._allocate(seq_group,
                                                      layerwise=True)
                if ret_status == 1:
                    break
                time.sleep(1 / 5e4)
                if show_sleep:
                    logger.info(f"Waiting for pre-allocated cache blocks for "
                                f"{seq_group.request_id}...")
                    show_sleep = False
            assert ret_status == 1, f"Cannot find pre-allocated cache blocks " \
                f"for {seq_group.request_id}! Return status: {ret_status}."
            self.scheduler.running.append(seq_group)

    @log_gpu_memory_usage(show_log=False, empty_cache=False, message="pipeline step")
    def pipeline_step(
        self,
        batch_request_ids: Optional[List[int]] = None,
        batch_output_tokens: Optional[List[int]] = None
    ) -> Tuple[Union[List[torch.Tensor], List[int]], List[List[int]]]:
        preempted_reqs = []
        if self.stage_id == 0:
            if self.tp_rank == 0:
                batch = self.prepare_inputs(batch_request_ids,
                                            batch_output_tokens)
                (input_tensor, input_positions, input_metadata, batch_reqs,
                 preempted_reqs) = batch
                logger.info(
                    f"Step {self.global_steps} BS {len(batch_reqs)} scheduled "
                    f"{input_tensor.size()} tokens, "
                    f"requests: {batch_request_ids}, preempted: {preempted_reqs}"
                )

                if len(batch_reqs) == 0:
                    return [], batch_reqs, batch_request_ids

                self.scatter_input(input_tensor, input_positions,
                                   input_metadata, batch_reqs)
            else:
                batch = self.gather_input()
                input_tensor, input_positions, input_metadata, batch_reqs = batch
        else:
            (input_tensor, input_positions, input_metadata,
             batch_reqs) = self.recv()

        output_tensor = self._run_workers("pure_forward",
                                          get_all_outputs=False,
                                          input_tensor=input_tensor,
                                          positions=input_positions,
                                          input_metadata=input_metadata)
        self.global_steps += 1

        if self.stage_id != self.num_stages - 1:
            self.send(output_tensor, input_positions, input_metadata,
                      batch_reqs)

        torch.cuda.synchronize()
        return output_tensor, batch_reqs, preempted_reqs

    def prepare_inputs(self, batch_request_ids: List[int],
                       batch_output_tokens: List[int]):
        batch_block_request_info = self.batch_scheduler.try_batch(
            batch_request_ids, batch_output_tokens)
        total_blocks_needed = batch_block_request_info[-1]
        batch_block_request = batch_block_request_info[:-1]
        if total_blocks_needed > 0:
            self.tcp_client.send_pyobj(
                ["cache_alloc", [self.model_name, batch_block_request]])
            block_info = self.tcp_client.recv_pyobj()
        else:
            block_info = np.array([len(batch_request_ids)], dtype=np.int32)

        batch_info = self.batch_scheduler.get_batch_info(
            batch_block_request, block_info)
        num_tokens = batch_info[0]
        num_contexts = batch_info[1]
        max_context_len = batch_info[2]
        max_num_blocks_per_seq = batch_info[3]
        num_layers = batch_info[4]
        num_heads = batch_info[5]
        prompt_lens = batch_info[6:]

        padded_len = (num_tokens + 7) // 8 * 8
        slot_mapping = torch.empty((num_tokens, num_layers, num_heads),
                                   dtype=torch.int64)
        if max_context_len > 0:
            block_tables = torch.empty(
                (num_tokens, max_num_blocks_per_seq, num_layers, num_heads),
                dtype=torch.int32)
        else:
            block_tables = torch.empty((0, max_num_blocks_per_seq, 0, 0),
                                       dtype=torch.int32)
        input_tensor = torch.empty((padded_len, ), dtype=torch.int32)
        input_positions = torch.empty((padded_len, ), dtype=torch.int32)
        context_lens = torch.empty((num_contexts, ), dtype=torch.int32)

        self.batch_scheduler.get_batch(batch_block_request, block_info,
                                       input_tensor, input_positions,
                                       context_lens, block_tables,
                                       slot_mapping)

        preempted_reqs = self.batch_scheduler.get_preempt_requests()
        batch_reqs = self.batch_scheduler.get_batch_reqs()

        slot_mapping = slot_mapping.cuda(non_blocking=True)
        block_tables = block_tables.cuda(non_blocking=True)
        input_tensor = input_tensor.cuda(non_blocking=True).long()
        input_positions = input_positions.cuda(non_blocking=True).long()
        context_lens = context_lens.cuda(non_blocking=True)
        block_tables = block_tables.permute(2, 0, 3, 1).contiguous()

        input_metadata = self._run_workers("_prepare_inputs_metadata",
                                           prompt_lens=prompt_lens,
                                           slot_mapping=slot_mapping,
                                           context_lens=context_lens,
                                           max_context_len=max_context_len,
                                           block_tables=block_tables)

        if preempted_reqs:
            dump_to_shared_var(self.preempt_req_shm_name, preempted_reqs)

        return input_tensor, input_positions, input_metadata, batch_reqs, preempted_reqs

    def _init_workers(self, stage_id: int, model_id: int):
        from muxserve.flexserver.pipeworker import PipeWorker

        self.workers: List[PipeWorker] = []
        worker = PipeWorker(self.model_config, self.parallel_config,
                            self.scheduler_config, 0, None,
                            self.model_config.flexstore_port, stage_id,
                            model_id)
        self.workers.append(worker)
        self._run_workers("init_model")

    def scatter_input(self, input_tensor: torch.Tensor,
                      positions: torch.Tensor, input_metadata: InputMetadata,
                      batch_reqs: List[List[int]]) -> None:
        # walkaround for cuda synchronization
        num_tokens = input_metadata.num_valid_tokens
        num_generation_tokens = input_metadata.num_generation_tokens
        num_prompts = input_metadata.num_prompts
        max_context_len = input_metadata.max_context_len
        prompt_lens = input_metadata.prompt_lens

        data = [
            num_tokens, num_generation_tokens, num_prompts, max_context_len
        ]
        data += prompt_lens
        for req in batch_reqs:
            data += req

        src = parallel_state.get_tensor_model_parallel_src_rank()
        group = parallel_state.get_tensor_model_parallel_group()
        if self.world_size > 8:
            batch_info = torch.tensor(data,
                                      dtype=torch.int32,
                                      device=torch.cuda.current_device())
            shape_tensor = torch.tensor([len(data)],
                                        dtype=torch.int32,
                                        device=torch.cuda.current_device())
            torch.distributed.broadcast(shape_tensor, src=src, group=group)
            torch.distributed.broadcast(batch_info, src=src, group=group)
        else:
            for shm_name in self.tp_share_name:
                while True:
                    try:
                        dump_to_shared_var(shm_name, data)
                        break
                    except FileExistsError:
                        time.sleep(1 / 5e4)

        torch.distributed.broadcast(input_tensor, src=src, group=group)
        torch.distributed.broadcast(positions, src=src, group=group)
        torch.distributed.broadcast(input_metadata.slot_mapping,
                                    src=src,
                                    group=group)
        if input_metadata.num_generation_tokens > 0:
            torch.distributed.broadcast(input_metadata.context_lens,
                                        src=src,
                                        group=group)
            torch.distributed.broadcast(input_metadata.block_tables,
                                        src=src,
                                        group=group)

    def gather_input(self) -> None:
        # walkaround for cuda synchronization
        src = parallel_state.get_tensor_model_parallel_src_rank()
        group = parallel_state.get_tensor_model_parallel_group()
        if self.world_size > 8:
            shape_tensor = torch.empty((1, ),
                                       dtype=torch.int32,
                                       device=torch.cuda.current_device())
            torch.distributed.broadcast(shape_tensor, src=src, group=group)
            batch_info = torch.empty((shape_tensor[0], ),
                                     dtype=torch.int32,
                                     device=torch.cuda.current_device())
            torch.distributed.broadcast(batch_info, src=src, group=group)
            data = batch_info.tolist()
        else:
            while True:
                data = load_from_shared_var(self.tp_share_name)
                if data:
                    break
                time.sleep(1 / 5e4)
        num_tokens = data.pop(0)
        num_generation_tokens = data.pop(0)
        num_prompts = data.pop(0)
        max_context_len = data.pop(0)
        prompt_lens = data[:num_prompts]
        batch_reqs = [[data[i], data[i + 1]]
                      for i in range(num_prompts, len(data), 2)]

        num_requests = len(batch_reqs)
        num_blocks = (max_context_len + self.block_size - 1) // self.block_size

        multiple_of = 8
        padded_num_tokens = (num_tokens + multiple_of -
                             1) // multiple_of * multiple_of
        input_tensor_shape = (padded_num_tokens, )
        positions_shape = (padded_num_tokens, )
        context_lens_shape = (num_generation_tokens, )
        batch_reqs_shape = (num_requests, 2)
        request_info_shape = (len(prompt_lens) + 1, )
        if self.is_muxserve:
            slot_mapping_shape = (num_tokens, self.num_layers, self.num_heads)
            block_tables_shape = (self.num_layers, num_generation_tokens,
                                  self.num_heads, num_blocks)
        else:
            slot_mapping_shape = (num_tokens, )
            block_tables_shape = (num_generation_tokens, num_blocks)

        device = torch.cuda.current_device()
        input_tensor = torch.empty(input_tensor_shape,
                                   dtype=torch.int64,
                                   device=device)
        positions = torch.empty(positions_shape,
                                dtype=torch.int64,
                                device=device)
        slot_mapping = torch.empty(slot_mapping_shape,
                                   dtype=torch.int64,
                                   device=device)
        context_lens = torch.empty(context_lens_shape,
                                   dtype=torch.int,
                                   device=device)
        block_tables = torch.empty(block_tables_shape,
                                   dtype=torch.int,
                                   device=device)

        torch.distributed.broadcast(input_tensor, src=src, group=group)
        torch.distributed.broadcast(positions, src=src, group=group)
        torch.distributed.broadcast(slot_mapping, src=src, group=group)
        if max_context_len > 0:
            torch.distributed.broadcast(context_lens, src=src, group=group)
            torch.distributed.broadcast(block_tables, src=src, group=group)

        input_metadata = self._run_workers("_prepare_inputs_metadata",
                                           prompt_lens=prompt_lens,
                                           slot_mapping=slot_mapping,
                                           context_lens=context_lens,
                                           max_context_len=max_context_len,
                                           block_tables=block_tables)

        return input_tensor, positions, input_metadata, batch_reqs

    def recv(self) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        # walkaround for cuda synchronization
        while True:
            data = load_from_shared_var(self.recv_shm_name)
            if data:
                break
            time.sleep(1 / 5e4)
        num_tokens = data.pop(0)
        num_generation_tokens = data.pop(0)
        num_prompts = data.pop(0)
        max_context_len = data.pop(0)
        prompt_lens = data[:num_prompts]
        batch_reqs = [[data[i], data[i + 1]]
                      for i in range(num_prompts, len(data), 2)]

        num_requests = len(batch_reqs)
        num_blocks = (max_context_len + self.block_size - 1) // self.block_size

        multiple_of = 8
        padded_num_tokens = (num_tokens + multiple_of -
                             1) // multiple_of * multiple_of
        input_tensor_shape = (padded_num_tokens,
                              self.model_config.get_hidden_size())
        positions_shape = (padded_num_tokens, )
        context_lens_shape = (num_generation_tokens, )
        batch_reqs_shape = (num_requests, 2)
        request_info_shape = (len(prompt_lens) + 1, )
        if self.is_muxserve:
            slot_mapping_shape = (num_tokens, self.num_layers, self.num_heads)
            block_tables_shape = (self.num_layers, num_generation_tokens,
                                  self.num_heads, num_blocks)
        else:
            slot_mapping_shape = (num_tokens, )
            block_tables_shape = (num_generation_tokens, num_blocks)

        self.pipeline_config.variable_seq_lengths = False
        old_pipeline_dtype = self.pipeline_config.pipeline_dtype
        input_tensor = recv_forward(input_tensor_shape, self.pipeline_config)

        self.pipeline_config.pipeline_dtype = torch.int64
        positions = recv_forward(positions_shape, self.pipeline_config)

        slot_mapping = recv_forward(slot_mapping_shape, self.pipeline_config)
        if max_context_len > 0:
            self.pipeline_config.pipeline_dtype = torch.int
            context_lens = recv_forward(context_lens_shape,
                                        self.pipeline_config)
            block_tables = recv_forward(block_tables_shape,
                                        self.pipeline_config)
        else:
            context_lens = torch.empty(context_lens_shape, dtype=torch.int)
            block_tables = torch.empty(block_tables_shape, dtype=torch.int)
        self.pipeline_config.pipeline_dtype = old_pipeline_dtype

        input_metadata = self._run_workers("_prepare_inputs_metadata",
                                           prompt_lens=prompt_lens,
                                           slot_mapping=slot_mapping,
                                           context_lens=context_lens,
                                           max_context_len=max_context_len,
                                           block_tables=block_tables)
        return input_tensor, positions, input_metadata, batch_reqs

    def send(self, output_tensor: torch.Tensor, positions: torch.Tensor,
             input_metadata: InputMetadata,
             batch_reqs: List[List[int]]) -> None:
        # walkaround for cuda synchronization
        num_tokens = input_metadata.num_valid_tokens
        num_generation_tokens = input_metadata.num_generation_tokens
        num_prompts = input_metadata.num_prompts
        max_context_len = input_metadata.max_context_len
        prompt_lens = input_metadata.prompt_lens

        data = [
            num_tokens, num_generation_tokens, num_prompts, max_context_len
        ]
        data += prompt_lens
        for req in batch_reqs:
            data += req
        while True:
            try:
                dump_to_shared_var(self.send_shm_name, data)
                break
            except FileExistsError:
                time.sleep(1 / 5e4)

        self.pipeline_config.variable_seq_lengths = False
        old_pipeline_dtype = self.pipeline_config.pipeline_dtype
        send_forward(output_tensor, self.pipeline_config)

        self.pipeline_config.pipeline_dtype = torch.int64
        send_forward(positions, self.pipeline_config)
        send_forward(input_metadata.slot_mapping, self.pipeline_config)
        if input_metadata.num_generation_tokens > 0:
            self.pipeline_config.pipeline_dtype = torch.int
            send_forward(input_metadata.context_lens, self.pipeline_config)
            send_forward(input_metadata.block_tables, self.pipeline_config)
        self.pipeline_config.pipeline_dtype = old_pipeline_dtype

    def release_request(self, free_request_ids: List[int],
                        finished_request_ids: List[int]):
        if finished_request_ids:
            self.tcp_client.send_pyobj(
                ["free_cache", [self.model_name, finished_request_ids]])
        self.batch_scheduler.release_requests(free_request_ids)
        if finished_request_ids:
            _ = self.tcp_client.recv_pyobj()

    def warmup(self) -> None:
        """Warms up the engine and cuda context."""
        prompt_tokens = [100] * 50
        max_tokens = 5
        batch_request_ids, batch_output_tokens = [], []
        if self.stage_id == 0 and self.tp_rank == 0:
            for i in range(16):
                req_id = next(self.unique_id)
                batch_request_ids.append(req_id)
                self.batch_scheduler.add_request(prompt_tokens, req_id,
                                                 max_tokens)

        generated_len = {}
        for step in range(5):
            if self.stage_id == 0:
                outputs, batch_reqs, _ = self.pipeline_step(
                    batch_request_ids, batch_output_tokens)

                if self.tp_rank == 0:
                    free_request_ids = []
                    batch_request_ids = []
                    batch_output_tokens = []
                    for (req_id, max_token) in batch_reqs:
                        if req_id not in generated_len:
                            generated_len[req_id] = 0
                        generated_len[req_id] += 1
                        if generated_len[req_id] == max_token:
                            free_request_ids.append(req_id)
                        else:
                            batch_request_ids.append(req_id)
                            batch_output_tokens.append(20)  # a random token

                    if free_request_ids:
                        self.release_request(free_request_ids,
                                             free_request_ids)
            else:
                outputs, batch_reqs, _ = self.pipeline_step()

        # wait for all requests to finish
        torch.distributed.all_reduce(
            torch.zeros(1).cuda(),
            group=parallel_state.get_tensor_model_parallel_group())
        torch.cuda.synchronize()

    def case_profile(self, batch_size: int, seq_len: int, is_prefill: bool):
        num_iters = 10 if is_prefill else 1
        max_tokens = 1 if is_prefill else 10
        prompt_tokens = np.random.randint(0,
                                          24000,
                                          (batch_size * num_iters, seq_len),
                                          dtype=np.int32)
        request_ids = []
        if self.stage_id == 0 and self.tp_rank == 0:
            for i in range(batch_size * num_iters):
                req_id = next(self.unique_id)
                request_ids.append(req_id)
                self.batch_scheduler.add_request(prompt_tokens[i], req_id,
                                                 max_tokens)

        if not is_prefill:
            # run prefill first
            max_num_seqs = 16384 // seq_len
            for i in range(0, batch_size, max_num_seqs):
                batch_request_ids = request_ids[i:i + max_num_seqs]
                if self.stage_id == 0:
                    outputs, batch_reqs, _ = self.pipeline_step(
                        batch_request_ids, [])
                else:
                    outputs, batch_reqs, _ = self.pipeline_step()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for i in range(num_iters):
            generated_len = {}
            batch_request_ids = request_ids[i * batch_size:(i + 1) *
                                            batch_size]
            batch_output_tokens = []
            if not is_prefill:
                batch_output_tokens = [20] * len(batch_request_ids)
            for step in range(max_tokens):
                if self.stage_id == 0:
                    outputs, batch_reqs, _ = self.pipeline_step(
                        batch_request_ids, batch_output_tokens)

                    if self.tp_rank == 0:
                        free_request_ids = []
                        batch_request_ids = []
                        batch_output_tokens = []
                        for (req_id, max_token) in batch_reqs:
                            if req_id not in generated_len:
                                generated_len[req_id] = 0
                            generated_len[req_id] += 1
                            if generated_len[req_id] == max_token:
                                free_request_ids.append(req_id)
                            else:
                                batch_request_ids.append(req_id)
                                # append a random token
                                batch_output_tokens.append(20)

                        if free_request_ids:
                            self.release_request(free_request_ids,
                                                 free_request_ids)
                else:
                    outputs, batch_reqs, _ = self.pipeline_step()

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_iters = num_iters if is_prefill else max_tokens
        avg_lat = (end_time - start_time) * 1000 / total_iters
        return avg_lat

    def profile_memory(self):
        self.scheduler_config.max_num_batched_tokens

    def profile(self):
        logger.info("Profiling...")
        log_str = "BS, SL, Prefill Lat (ms), Decode Lat (ms)\n"
        num_gpu_blocks = self.cache_config.num_gpu_blocks

        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        max_batch_size = max(batch_sizes) * self.world_size
        while True:
            if batch_sizes[-1] * 2 <= max_batch_size:
                batch_sizes.append(batch_sizes[-1] * 2)
            else:
                break

        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        vocab_size = self.model_config.hf_config.vocab_size
        prefill_max_num_batched_tokens = int(free_gpu_memory // 2 //
                                             vocab_size // 4)
        logger.info(f"Max batched tokens: {prefill_max_num_batched_tokens}")
        decoding_max_num_batched_tokens = np.inf
        for bs in batch_sizes:
            max_seq_len = (num_gpu_blocks //
                           (self.num_heads * self.num_layers) //
                           bs) * self.block_size
            for sl in [32, 64, 128, 256, 512]:
                if sl + 10 > max_seq_len:
                    log_str += f"{bs}, {sl}, N/A, N/A\n"
                    continue
                if bs < decoding_max_num_batched_tokens:
                    decode_avg_lat = self.case_profile(bs, sl, False)
                    if decode_avg_lat >= 1_000:
                        decoding_max_num_batched_tokens = min(
                            decoding_max_num_batched_tokens, bs)
                        prefill_max_num_batched_tokens = min(
                            prefill_max_num_batched_tokens, bs * sl)
                else:
                    decode_avg_lat = -1

                if bs * sl < prefill_max_num_batched_tokens:
                    prefill_avg_lat = self.case_profile(bs, sl, True)
                    if prefill_avg_lat >= 5_000:
                        prefill_max_num_batched_tokens = min(
                            prefill_max_num_batched_tokens, bs * sl)
                else:
                    prefill_avg_lat = -1
                log_str += f"{bs}, {sl}, {prefill_avg_lat:.3f}, " \
                           f"{decode_avg_lat:.3f}\n"
        if self.tp_rank == 0:
            print("Profiling results:")
            print(log_str)

    def torch_profiler(self, batch_size: int, seq_len: int):
        with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
        ],
                                    with_stack=True,
                                    with_modules=True) as prof:
            self.case_profile(batch_size, seq_len, True)
            self.case_profile(batch_size, seq_len, False)
        model = self.model_config.model.split("/")[-1]
        out_trace = f"log/profiler_muxserve/{model}_bs{batch_size}_seq{seq_len}_rank{self.tp_rank}.json"
        logger.info(f"Export profiler trace to {out_trace}")
        prof.export_chrome_trace(out_trace)

    def _run_workers(
        self,
        method_name: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""

        assert len(self.workers) == 1, "worker only 1 for pipeline vllm"
        all_outputs = []
        for worker in self.workers:
            executor = getattr(worker, method_name)
            output = executor(*args, **kwargs)
            all_outputs.append(output)

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output
