import os
import copy
import time
import torch
import numpy as np
import math
import re
from typing import Optional

from transformers import AutoConfig, PretrainedConfig
from typing import List, Tuple, Iterator, Dict, Union
from muxserve.config import MuxServeConfig, JobConfig
from muxserve.logger import get_logger
from muxserve.flexstore.weight_utils import hf_model_weights_iterator
from muxserve.zmq_utils import ZMQServer

from muxserve.memory_manager import KVStorage
from muxserve.flexserver.pipeworker import PipeWorker

logger = get_logger()

KVCache = Tuple[torch.Tensor, torch.Tensor]


def replace_numbers_with_value(s: str, replacement_value: int):
    regex_pattern = r'\d+'
    result = re.sub(regex_pattern, str(replacement_value), s)
    return result


def grasp_num(s: str):
    regex_pattern = r'\d+'
    result = re.findall(regex_pattern, s)
    return int(result[0])


class WeightStorage:

    # FIXME: Add a parameter to specify the dataType, maybe as a filed of `JobConfig`
    def __init__(self, weights: Dict[str, torch.Tensor], job_config: JobConfig,
                 model_config: PretrainedConfig, rank_start: int,
                 rank_end: int) -> None:
        # FIXME: dp is not taken into consideration
        placement = job_config.placement[0]
        self.dtype = job_config.model_dtype
        logger.info(
            f"load_weight: {job_config.model}; placement, dtype: {placement}, {self.dtype}"
        )
        # for world_size `P`, we have {weight_name: [weight_1, weight_2, ..., weight_P]}
        self.data: Dict[int, Dict[str,
                                  torch.Tensor]] = {k: {}
                                                    for k in placement}
        # meta_data to rebuild tensors; {rank: {weight_name: meta_data}}
        self.metadata: Dict[int, Dict[str, Dict]] = {k: {} for k in placement}

        reshaped_weights = WeightStorage.reshape_weights(
            weights, job_config, model_config)
        tp_size = job_config.tensor_parallel_size
        pp_size = job_config.pipeline_parallel_size
        if model_config.model_type == "llama":
            '''
            VocabParallelEmbedding: column [vocab, hidden]
            lm_head: column [vocab, hidden]

            input_layernorm: no tp
            post_attn_layernorm: no tp
            MLP:
                gate_up_proj: column [interm*2, hidden]
                down_proj: row [hidden, interm]
            Attn:
                qkv_proj: column [head_dim*(num_q_heads+num_kv_heads), hidden]
                o_proj: row [num_heads*head_dim, hidden]
            '''
            # for tensor parallel
            col_split = ["qkv_proj", "gate_up_proj", "embed_tokens", "lm_head"]
            row_split = ["down_proj", "o_proj"]

            # for pipeline parallel
            pre_process_weights = ["model.embed_tokens.weight"]
            post_process_weights = ["model.norm.weight", "lm_head.weight"]
            # pipeline_partition to save the num of hidden layer of every pipeline stage
            pipeline_partition: List[int] = PipeWorker.pipeline_split(
                model_config.num_hidden_layers, pp_size)

        else:
            raise RuntimeError("Only Llama supported now")

        logger.info(f"### Begin to place weights on GPUs ...")
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for name, val in reshaped_weights.items():
            is_column_parallel = False
            for p in col_split:
                if p in name:
                    shard_size = val.shape[0] // tp_size
                    for idx, dev_idx in enumerate(placement):
                        if dev_idx < rank_start or dev_idx >= rank_end:
                            continue
                        local_rank = dev_idx - rank_start
                        pp_rank = idx // tp_size
                        tp_rank = idx % tp_size

                        pre_process = (pp_rank == 0)
                        post_process = (pp_rank == pp_size - 1)

                        # for embed_tokens
                        if name in pre_process_weights:
                            mapped_name = name
                            if not pre_process:
                                continue
                        elif name in post_process_weights:
                            mapped_name = name
                            if not post_process:
                                continue
                        else:
                            original_layer_idx = grasp_num(name)
                            placed_stage = -1
                            while original_layer_idx >= 0:
                                mapped_layer_idx = original_layer_idx
                                placed_stage += 1
                                original_layer_idx -= pipeline_partition[
                                    placed_stage]
                            if placed_stage != pp_rank:
                                continue
                            mapped_name = replace_numbers_with_value(
                                name, mapped_layer_idx)

                        # split along the 1st dim
                        weight = val[tp_rank * shard_size:(tp_rank + 1) *
                                     shard_size]
                        self.data[dev_idx][mapped_name] = weight.to(
                            f"cuda:{local_rank}", dtype=self.dtype)

                    is_column_parallel = True
                    break
            if is_column_parallel:
                continue

            is_row_parallel = False
            for p in row_split:
                if p in name:
                    shard_size = val.shape[1] // tp_size
                    for idx, dev_idx in enumerate(placement):
                        if dev_idx < rank_start or dev_idx >= rank_end:
                            continue
                        local_rank = dev_idx - rank_start
                        pp_rank = idx // tp_size
                        tp_rank = idx % tp_size
                        original_layer_idx = grasp_num(name)
                        placed_stage = -1
                        while original_layer_idx >= 0:
                            mapped_layer_idx = original_layer_idx
                            placed_stage += 1
                            original_layer_idx -= pipeline_partition[
                                placed_stage]
                        if placed_stage != pp_rank:
                            continue
                        mapped_name = replace_numbers_with_value(
                            name, mapped_layer_idx)

                        # split along the 2nd dim
                        weight = val[:, tp_rank * shard_size:(tp_rank + 1) *
                                     shard_size]
                        self.data[dev_idx][mapped_name] = weight.to(
                            f"cuda:{local_rank}", dtype=self.dtype)
                    is_row_parallel = True
                    break
            if is_row_parallel:
                continue

            # Otherwise, replicate the weight to each devices
            for idx, dev_idx in enumerate(placement):
                if dev_idx < rank_start or dev_idx >= rank_end:
                    continue
                local_rank = dev_idx - rank_start
                pp_rank = idx // tp_size
                pre_process = (pp_rank == 0)
                post_process = (pp_rank == ((len(placement) // tp_size) - 1))

                if name in post_process_weights:
                    if not post_process:
                        continue

                self.data[dev_idx][name] = val.to(f"cuda:{local_rank}",
                                                  dtype=self.dtype)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        logger.info(f"### Cost of data transfer(cpu->gpu): {t2-t1:.3f} s")

        for dev_id, weight_info in self.data.items():
            for weight_name, weight_val in weight_info.items():
                self.metadata[dev_id][weight_name] = get_tensor_metadata(
                    weight_val)

    def __str__(self) -> str:
        res = "{\n"
        for rank, weight_info in self.data.items():
            for weight_name, weight_val in weight_info.items():
                res += f"  {weight_name}_rank_{rank}: {weight_val.shape}; {weight_val.device}\n"
        return res + "}\n"

    @classmethod
    def from_iter(cls, iter: Iterator[Tuple[str, torch.Tensor]],
                  job_config: JobConfig, model_config: PretrainedConfig,
                  rank_start: int, rank_end: int):
        return cls({layer_name: data
                    for (layer_name, data) in iter}, job_config, model_config,
                   rank_start, rank_end)

    @staticmethod
    def reshape_weights(weights: Dict[str, torch.Tensor],
                        job_config: JobConfig, model_config: PretrainedConfig):
        '''
        For Llama, this method will cat [w_q,w_k,w_v] and [w_gate, w_up]
        '''
        tp_size = job_config.tensor_parallel_size
        weights_copy = copy.deepcopy(weights)

        if model_config.model_type == "llama":
            q_proj_shard_size = (model_config.hidden_size // tp_size)
            num_kv_heads_replicas = max(
                1, tp_size // model_config.num_key_value_heads)
            num_kv_heads_per_gpu = max(
                1, model_config.num_key_value_heads // tp_size)
            kv_proj_shard_size = (model_config.hidden_size //
                                  model_config.num_attention_heads *
                                  num_kv_heads_per_gpu)
            attn_weight_specs = [
                # (weight_name, shard_size, offset)
                ("q_proj", q_proj_shard_size, 0),
                ("k_proj", kv_proj_shard_size, q_proj_shard_size),
                ("v_proj", kv_proj_shard_size,
                 q_proj_shard_size + kv_proj_shard_size),
            ]
            per_rank_qkv_proj_size = q_proj_shard_size + kv_proj_shard_size * 2
            gate_up_shard_size = model_config.intermediate_size // tp_size
            per_rank_gate_up_proj_size = gate_up_shard_size * 2
            # print(f"tp_size: {tp_size}")
            # print(f"q_proj_shard_size: {q_proj_shard_size}")
            # print(f"kv_proj_shard_size: {kv_proj_shard_size}")
            # print(f"per_rank_qkv_proj_size: {per_rank_qkv_proj_size}")
            # print(f"per_rank_gate_up_proj_size: {per_rank_gate_up_proj_size}")
            for name, loaded_weight in weights.items():
                if "rotary_emb.inv_freq" in name:
                    del weights_copy[name]
                    continue

                # q_proj: [q_num_heads*head_dim, hidden], k/v_proj: [k/v_num_heads*head_dim, hidden]
                # cat qkv along the 1st dim: [(2*kv_num_heads+q_num_heads)*head_dim, hidden]
                is_attn_weight = False
                for wname, shard_size, offset in attn_weight_specs:
                    if wname not in name:
                        continue
                    cat_weight_name = name.replace(wname, "qkv_proj")
                    # 1. initialization
                    if cat_weight_name not in weights_copy:
                        weights_copy[cat_weight_name] = torch.empty(
                            ((q_proj_shard_size + 2 * kv_proj_shard_size) *
                             tp_size, model_config.hidden_size),
                            dtype=torch.float16)

                    # 2. copy values
                    cat_weight = weights_copy[cat_weight_name]
                    for tp_rank in range(tp_size):
                        if wname in ["k_proj", "v_proj"]:
                            shard_id = tp_rank // num_kv_heads_replicas
                        else:
                            shard_id = tp_rank
                        # print(
                        #     f"### copy from {name}: [{shard_size * shard_id}:{shard_size*(shard_id+1)}] to "
                        #     f"{cat_weight_name} [{offset+(tp_rank*per_rank_qkv_proj_size)}:"
                        #     f"{offset+shard_size+(tp_rank*per_rank_qkv_proj_size)}]\n"
                        # )
                        cat_weight[offset +
                                   (tp_rank * per_rank_qkv_proj_size):offset +
                                   shard_size +
                                   (tp_rank * per_rank_qkv_proj_size)].copy_(
                                       loaded_weight[shard_size *
                                                     shard_id:shard_size *
                                                     (shard_id + 1)])
                    del weights_copy[name]
                    is_attn_weight = True
                    break
                if is_attn_weight:
                    continue

                # gate_proj: [intermediate_size, hidden], up_proj: [intermediate_size, hidden]
                # cat `gate_proj` and `up_proj` along the 1st dim
                is_gate_up_weight = False
                for stride_id, wname in enumerate(["gate_proj", "up_proj"]):
                    if wname not in name:
                        continue
                    cat_weight_name = name.replace(wname, "gate_up_proj")
                    # 1. initialization
                    if cat_weight_name not in weights_copy:
                        weights_copy[cat_weight_name] = torch.empty(
                            (loaded_weight.shape[0] * 2,
                             loaded_weight.shape[1]),
                            dtype=torch.float16)

                    # 2. copy values
                    cat_weight = weights_copy[cat_weight_name]
                    shard_size = gate_up_shard_size
                    for tp_rank in range(tp_size):
                        # print(
                        #     f"### copy from {name}: [{shard_size * tp_rank}:{shard_size*(tp_rank+1)}] to "
                        #     f"{cat_weight_name} [{stride_id*shard_size + tp_rank*shard_size*2}:"
                        #     f"{(stride_id+1)*shard_size+tp_rank*shard_size*2}]\n"
                        # )
                        cat_weight[stride_id * shard_size +
                                   (tp_rank * per_rank_gate_up_proj_size):
                                   (stride_id + 1) * shard_size +
                                   (tp_rank *
                                    per_rank_gate_up_proj_size)].copy_(
                                        loaded_weight[shard_size *
                                                      tp_rank:shard_size *
                                                      (tp_rank + 1)])
                    del weights_copy[name]
                    is_gate_up_weight = True
                    break
                if is_gate_up_weight:
                    continue
        else:
            raise RuntimeError("Only Llama supported now")

        return weights_copy


class FlexStoreManager:
    """Manage the memory space across multiple GPUs."""

    def __init__(self, muxserve_config: MuxServeConfig):
        self.config = muxserve_config
        self.port = self.config.flexstore_port

        # FIXME: we use 128 currently, but it should be configurable.
        self.head_size = 128
        self.block_size = self.config.block_size

        use_openmpi = os.environ.get("OMPI_COMM_WORLD_SIZE", None) is not None
        use_mpich = os.environ.get("PMI_SIZE", None) is not None
        if use_openmpi:
            local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
            local_world_size = int(
                os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', 1))
            rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
            world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
        elif use_mpich:
            local_rank = int(os.environ.get('MPI_LOCALRANKID', 0))
            local_world_size = int(os.environ.get('MPI_LOCALNRANKS', 1))
            rank = int(os.environ.get('PMI_RANK', 0))
            world_size = int(os.environ.get('PMI_SIZE', 1))
        else:
            # local_rank = int(os.environ.get('LOCAL_RANK', 0))
            # local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
            # rank = int(os.environ.get('RANK', 0))
            # world_size = int(os.environ.get('WORLD_SIZE', 1))
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            local_world_size = self.config.nproc_per_node
            rank = self.config.node_rank
            world_size = self.config.nnodes

        self.local_world_size = local_world_size
        # convert #NODEs to #GPUs
        self.world_size = world_size * self.local_world_size
        self.rank = rank
        self.rank_start = rank * self.local_world_size
        self.rank_end = (rank + 1) * self.local_world_size

        # get all devices
        devices: set[int] = set()
        # map model_rank to real device_id
        self.rank_to_dev: Dict[str, List[int]] = {}
        for job in self.config.job_configs:
            self.rank_to_dev[job.name] = []
            # FIXME: dp is not taken into consideration
            for dev_id in job.placement[0]:
                devices.add(dev_id)
                self.rank_to_dev[job.name].append(dev_id)
        self.devices = list(devices)
        logger.info(f"rank_to_dev: {self.rank_to_dev}")

        # load all the deployed model weights
        self.models_weights: Dict[str, WeightStorage] = self.load_models()
        self.memory_stats("After loading models")

        # The physical cache blocks for all the models
        self.gpu_cache: Dict[int, KVCache] = self.allocate_gpu_cache()

        # for client to rebuild cache
        self.gpu_cache_matedata: Dict[int, Dict] = {
            k: (get_tensor_metadata(v[0]), get_tensor_metadata(v[1]))
            for (k, v) in self.gpu_cache.items()
        }

        # block manager
        num_total_blocks = self.get_num_gpu_blocks()
        self.block_manager = KVStorage(num_total_blocks)
        self.model_cache_info: Dict[str, Tuple[int, int]] = {}
        self.model_occupy_info: Dict[str, int] = {}
        for job_config in self.config.job_configs:
            model_name = job_config.name
            model_config = AutoConfig.from_pretrained(job_config.model)
            tensor_parallel_size = job_config.tensor_parallel_size
            num_heads = model_config.num_attention_heads // tensor_parallel_size
            pipeline_partition = PipeWorker.pipeline_split(
                model_config.num_hidden_layers,
                job_config.pipeline_parallel_size)
            num_hidden_layers = max(pipeline_partition)
            self.model_cache_info[model_name] = (num_hidden_layers, num_heads)
            self.model_occupy_info[model_name] = 0
        self.memory_stats("After init cache view")

        # Store block_table from clients
        self.block_table_storage: Dict = {}

        # record blocks allocated for each request
        self.request_to_blocks: Dict[int, List[int]] = {}

    def load_models(self) -> Dict[str, WeightStorage]:
        res: Dict[str, WeightStorage] = {}
        job_configs = self.config.job_configs

        for config in job_configs:
            tp_size = config.tensor_parallel_size
            pp_size = config.pipeline_parallel_size
            dev_count = torch.cuda.device_count()
            assert self.world_size >= tp_size * pp_size
            assert all(
                len(placement) == tp_size or len(placement) == pp_size
                for placement in config.placement)

        for job_config in job_configs:
            logger.info(f"Rank {self.rank} starts loading {job_config.name} "
                        f"({job_config.model}) ...")
            weight_iter = hf_model_weights_iterator(job_config.model)
            model_config = AutoConfig.from_pretrained(job_config.model)
            res[job_config.name] = WeightStorage.from_iter(
                weight_iter, job_config, model_config, self.rank_start,
                self.rank_end)

        return res

    def get_num_gpu_blocks(self) -> int:
        # TODO: compute
        total_mem_each_gpu = get_gpu_memory()  # get info from gpu0
        # FIXME: set gpu_memory_utilization in shell script manually
        avaliable_mem_each_gpu = self.config.gpu_memory_utilization * total_mem_each_gpu
        avaliable_mem_each_gpu = round(avaliable_mem_each_gpu) - 1

        block_mem = 2 * (np.prod(self.get_key_block_shape()) +
                         np.prod(self.get_value_block_shape()))
        max_num_blocks = (avaliable_mem_each_gpu // block_mem // 128) * 128
        return max_num_blocks

    def get_key_block_shape(self, element_size=2) -> Tuple[int, int, int]:
        x = 16 // element_size
        return (
            self.head_size // x,
            self.block_size,
            x,
        )

    def get_value_block_shape(self) -> Tuple[int, int]:
        return (
            self.head_size,
            self.block_size,
        )

    def allocate_gpu_cache(self) -> Dict[int, KVCache]:
        '''
        return: {model_name: kv_cache_for_model }
        '''
        k_block_shape = self.get_key_block_shape()
        v_block_shape = self.get_value_block_shape()

        gpu_cache: Dict[int, KVCache] = dict.fromkeys(self.devices)

        for device_id in self.devices:
            if device_id < self.rank_start or device_id >= self.rank_end:
                gpu_cache.pop(device_id)
                continue
            local_rank = device_id - self.rank_start
            num_gpu_blocks = self.get_num_gpu_blocks()
            k_cache = torch.empty(
                size=(num_gpu_blocks, *k_block_shape),
                dtype=torch.float16,  # TODO: configure it
                device=f"cuda:{local_rank}")
            v_cache = torch.empty(
                size=(num_gpu_blocks, *v_block_shape),
                dtype=torch.float16,  # TODO: configure it
                device=f"cuda:{local_rank}")
            gpu_cache[device_id] = (k_cache, v_cache)
            logger.info(f"Allocate {num_gpu_blocks} blocks "
                        f"({2*k_cache.nelement()*2/1e9} GB) KV Cache "
                        f"on cuda:{local_rank}")

        return gpu_cache

    def init_cache_view(self) -> Dict[str, KVStorage]:
        job_configs = self.config.job_configs
        cache_view: Dict[str, KVStorage] = {}
        num_total_blocks = self.get_num_gpu_blocks()
        logger.info(f">>> cache memory manager has {num_total_blocks} blocks")
        # TODO: here is some blocks waste,
        # assume we have 10000 `num_total_blocks`,
        # llama-7b and llama-13b has `group_size` = `num_heads*hidden_layers`, i.e. 32*32 and 40*40,
        # then we will:
        # 1. alloc `(10000 // (32*32+40*40)) = 3` block_group for each model.
        #   To make attention operator happy, we should align to `group_size` for each model.
        #   alloc for llama-7b: 3*32*32 = 3072: (0, 1024, 2048)
        #   start index for llama-13b is `math.ceil(3072/40)*40` = 3080: (3080, 4680, 6280)
        #   blocks in [3072, 3080) are wasted
        # 2. alloc remaining [7880, 10000) with dynamic programing to minimize waste.
        #   now we simply assign to the last model...

        model_names = []
        model_num_heads = []
        model_group_size = []
        model_placement = []
        for job_config in job_configs:
            model_config = AutoConfig.from_pretrained(job_config.model)
            tensor_parallel_size = job_config.tensor_parallel_size
            num_heads = model_config.num_attention_heads // tensor_parallel_size
            partition = PipeWorker.pipeline_split(
                model_config.num_hidden_layers,
                job_config.pipeline_parallel_size)
            num_hidden_layers = max(partition)

            model_num_heads.append(num_heads)
            model_group_size.append(num_heads * num_hidden_layers)
            model_names.append(job_config.name)
            # FIXME: dp is not taken into consideration
            model_placement.append(job_config.placement[0])

        # {dev => [(model_name, group_size, num_heads), ...]}
        dev_to_models: Dict[int, List[Tuple[str, int, int]]] = {}
        for k in self.devices:
            info = []
            for i, name in enumerate(model_names):
                if k in model_placement[i]:
                    info.append(
                        (name, model_group_size[i], model_num_heads[i]))
            dev_to_models[k] = info
        # {model_name => {dev => free_indices}}
        avaliable_groups: Dict[str, Dict[int, List[int]]] = {
            name: {dev: []
                   for dev in model_placement[i]}
            for (i, name) in enumerate(model_names)
        }
        global_offset: Dict[int, int] = {k: 0 for k in self.devices}

        for dev, models in dev_to_models.items():
            # 1. alloc for each model
            num_groups_each_model = (num_total_blocks - sum(
                m[1] for m in models)) // sum(m[1] for m in models)
            for (name, group_size, num_heads) in models:
                # To make attention operator happy, align to `group_size`
                global_offset[dev] = math.ceil(
                    global_offset[dev] / group_size) * group_size
                for _ in range(num_groups_each_model):
                    avaliable_groups[name][dev].append(global_offset[dev])
                    global_offset[dev] += group_size
            # 2. alloc remaining blocks
            start = global_offset[dev]
            name = models[-1][0]
            for idx in range(start, num_total_blocks - group_size + 1,
                             group_size):
                avaliable_groups[name][dev].append(idx)
        for name, num_heads, group_size in zip(model_names, model_num_heads,
                                               model_group_size):
            cache_view[name] = KVStorage(name, num_heads, group_size,
                                         self.global_free_blocks,
                                         avaliable_groups[name])
            logger.info(f">>>   Init cache view for {name}:")
            logger.info(
                f">>>     num_heads: {num_heads}, blocks per group: {group_size}"
            )
            for dev in self.devices:
                num_groups = len(avaliable_groups[name][dev])
                logger.info(f">>>     Dev {dev}: {num_groups} groups")

        return cache_view

    def get_free_block_info(self,
                            device_id) -> List[Tuple[str, Tuple[int, int]]]:
        '''
        return: [(model_name, (free_block_start_index, num_free_blocks))]
        '''
        res = []  # {model_name: avaliable_blocks }
        job_configs = self.config.job_configs
        for job in job_configs:
            name = job.name
            cache_view = self.cache_view[name]
            res.append((name, cache_view.find_consecutive_blocks(device_id)))
        return sorted(res, key=lambda x: x[1][1], reverse=True)

    def move_cache_blocks(self,
                          device_id: int,
                          dst: str,
                          src: Tuple[str, Tuple[int, int]],
                          alpha: float = 0.5):
        '''
        move partial kv_cache of model `src` to model `dst`
        src: [model_name, (start_block_index, num_blocks)]
            use (`start_block_index`,`num_blocks`) to repr consecutive free blocks of `src`
        dst: model_name of which kv_cache is extended
        alpha: `alpha * src.num_blocks` will be moved from `src` to `dst`
        '''
        src_name, (src_start, src_num_blocks) = src
        src_gsize = self.cache_view[src_name].group_size
        dst_gsize = self.cache_view[dst].group_size
        dst_heads = self.cache_view[dst].num_heads

        num_remain_blocks = math.floor((1 - alpha) * src_num_blocks)
        num_remain_blocks = num_remain_blocks // src_gsize * src_gsize
        remain_group_indices = [
            src_start + x for x in range(0, num_remain_blocks, src_gsize)
        ]
        src_free_group_indices = [
            src_start + x for x in range(0, src_num_blocks, src_gsize)
        ]
        delete_group_indices = set(src_free_group_indices) - set(
            remain_group_indices)

        # FIXME: now we simply discard the wasted blocks, we should trace these blocks
        taken_blocks_start_idx = src_start + num_remain_blocks
        # to make attention operator happy; align to `dst_gsize`
        aligned_taken_blocks_start_idx = math.ceil(
            taken_blocks_start_idx / dst_gsize) * dst_gsize
        num_taken_blocks = src_num_blocks - num_remain_blocks
        num_taken_blocks = num_taken_blocks - (aligned_taken_blocks_start_idx -
                                               taken_blocks_start_idx)
        taken_group_indices = [
            aligned_taken_blocks_start_idx + x
            for x in range(0, num_taken_blocks - dst_gsize + 1, dst_gsize)
        ]
        if len(taken_group_indices) <= 0:
            logger.info(f"No blocks can be taken from {src_name} to {dst}")
            return

        dst_avaliable_group = self.cache_view[dst].avaliable_groups[device_id]
        self.cache_view[dst].num_total_groups += len(taken_group_indices)

        dst_avaliable_group.extend(taken_group_indices)
        dst_avaliable_group = sorted(dst_avaliable_group)
        src_avaliable_group = self.cache_view[src_name].avaliable_groups[
            device_id]
        src_avaliable_group = sorted(
            list(set(src_avaliable_group) - delete_group_indices))
        self.cache_view[src_name].num_total_groups -= len(delete_group_indices)

        # update block info
        self.cache_view[dst].avaliable_groups[device_id] = dst_avaliable_group
        self.cache_view[src_name].avaliable_groups[
            device_id] = src_avaliable_group

        for idx in taken_group_indices:
            for i in range(dst_gsize):
                assert self.global_free_blocks[device_id][
                    idx + i], f"[{src_name}] block {idx} already occupied"

        waste_info = f"blocks: [{taken_blocks_start_idx}, {aligned_taken_blocks_start_idx}) are wasted" \
            if taken_blocks_start_idx != aligned_taken_blocks_start_idx \
            else "No waste in this reallocation"

        # logger.info(f"remain_group_indices: {remain_group_indices}")
        # logger.info(f"taken_group_indices: {taken_group_indices}")
        # logger.info(f"delete_group_indices: {delete_group_indices}")

        # for logger
        delete_group_indices = sorted(list(delete_group_indices))
        dst_after_move = self.cache_view[dst].get_num_free_groups(device_id)
        src_after_move = self.cache_view[src_name].get_num_free_groups(
            device_id)
        logger.info(
            f"realloc: KV Cache Reallocation is triggered\n"
            f"realloc: {src} -> {dst}; cuda:{device_id};\n"
            f"realloc: {len(taken_group_indices)} block_groups is moved   into {dst}, total {dst_after_move}; "
            f"[{taken_group_indices[0]} .. {taken_group_indices[-1]}]\n"
            f"realloc: {len(delete_group_indices)} block_groups is removed from {src[0]}, total {src_after_move}; "
            f"[{delete_group_indices[0]} .. {delete_group_indices[-1]}]\n"
            f"realloc: {waste_info}")

    @staticmethod
    def parse_request(req: Tuple):
        req_type = req[0]
        req_args = req[1]
        if req_type not in [
                "get_rank",
                "init_finished",
                "query_num_ready_processes",
                "get_num_blocks",
                "weight",
                "cache_init",
                "cache_alloc",
                "start_warmup",
                "warmup_ready",
                "blocktable_load",
                "blocktable_store",
                "free_cache",
                "lock_init",
                "log_stats",
                "exit",
        ]:
            return None
        return (req_type, req_args)

    def deploy(self):
        '''
        weight request format:
            Tuple["weight", [{rank}, {model_name}]]
            - memory_manager return:
                model_weight_on_{rank}

        cache_init request format:
            Tuple["cache_init", [{rank}, {model_name}]]
            - memory_manager return:
                (k_blocks, v_blocks)
                # length of the `k/v_blocks` is `num_total_blocks`

        cache_alloc request format:
            Tuple["cache_alloc", [{req_id}, {rank}, {model_name}, {num_req_groups}]]
            - memory_manager return:
                lead_free_block_idx, the subsequent `num_layers*num_heads` blocks are allocated

        blocktable_load request format:
            Tuple["blocktable_load", [{req_id}]]
            - memory_manager return:
                block_table which is stored by prefill process of this `req_id`

        blocktable_store request format:
            Tuple["blocktable_store", [{req_id}, {layer-wise-block_table}]]
            - memory_manager will store the layer-wise `blocktable`

        free_cache request format:
            Tuple["free_cache", [{rank}, {model_name}, {layer-wise-block_table}]]
            - memory_manager will collect cache blocks freed by this request
        '''
        tcp_server = ZMQServer("localhost", self.port)
        proc_id = 1
        proc_id_map = {}
        lock_tensor = torch.tensor(0, dtype=torch.int, device='cuda:0')
        num_realloc, total_realloc_time = 0, 0
        ready_processes = 0
        process_in_warmup = False

        logger.info(f"Memory manager is listening on {self.port} ...")
        while True:
            req = tcp_server.recv_pyobj()

            parse_res = FlexStoreManager.parse_request(req)

            if parse_res is None:
                logger.info(f"Recv incorrect format: {req}")
                tcp_server.send_pyobj("Incorrect format")
                continue

            req_type, req_args = parse_res

            if req_type == "get_rank":
                local_rank = req_args
                ret = self.rank * self.local_world_size + local_rank
            elif req_type == "init_finished":
                ret = ready_processes == self.config.num_runtime_processes
            elif req_type == "query_num_ready_processes":
                ret = ready_processes
            elif req_type == "get_num_blocks":
                ret = self.get_num_gpu_blocks()
            elif req_type == "start_warmup":
                if process_in_warmup:
                    ret = False
                else:
                    process_in_warmup = True
                    ret = True
            elif req_type == "warmup_ready":
                process_in_warmup = False
                ready_processes += 1
                ret = ready_processes == self.config.num_runtime_processes
                logger.info(
                    f"{ready_processes}/{self.config.num_runtime_processes} "
                    f"processes ready")
            elif req_type == "weight":
                logger.info(f"Receive {req_type}, {req_args}")
                rank, model_name = req_args
                req_weight = self.models_weights[model_name]
                dev_id = self.rank_to_dev[model_name][rank]
                ret = req_weight.metadata[dev_id]
            elif req_type == "cache_init":
                logger.info(f"Receive {req_type}, {req_args}")
                rank, model_name = req_args
                dev_id = self.rank_to_dev[model_name][rank]
                ret = self.gpu_cache_matedata[dev_id]
            elif req_type == "lock_init":
                logger.info(f"Receive {req_type}, {req_args}")
                rank, model_name, mps_percentage = req_args
                key = (model_name, mps_percentage)
                if key not in proc_id_map:
                    proc_id_map[key] = proc_id
                    proc_id += 1

                lock_meta_data = get_tensor_metadata(lock_tensor)
                ret = {
                    "lock_tensor": lock_meta_data,
                    "proc_id": proc_id_map[key]
                }
            elif req_type == "cache_alloc":
                model_name, batch_info = req_args
                num_layers, num_heads = self.model_cache_info[model_name]
                ret = self.block_manager.allocate_batch(
                    batch_info, num_layers, num_heads)
                # logger.info(
                #     f"cache_alloc: {ret}, {type(ret)} size: {ret.shape}")
            elif req_type == "free_cache":
                model_name, finished_request_ids = req_args
                num_freed_blocks = self.block_manager.free_batch(
                    finished_request_ids)
                self.model_occupy_info[model_name] -= num_freed_blocks
                ret = None
            elif req_type == "log_stats":
                # logger.info(self.inspect())
                logger.info("Receive log_stats")
                ret = None
            elif req_type == "exit":
                logger.info(f"Receive {req_type}, exit...")
                del self.models_weights
                break
            else:
                logger.warning(f"Receive {req_type}, {req_args}")
                raise RuntimeError("Unknown request type")

            tcp_server.send_pyobj(ret)

            # print cache usage for each LLM
            # if req_type == "cache_alloc":
            #     num_blocks = self.block_manager.get_new_blocks_allocated()
            #     self.model_occupy_info[model_name] += num_blocks
            # self.log_cache_usage()

    def log_cache_usage(self):
        logstr = "[Block Usage] "
        for model_name, num_blocks in self.model_occupy_info.items():
            logstr += f"{model_name}: {num_blocks} ,"
        logstr = logstr[:-2]
        logger.info(logstr)

    def inspect(self) -> str:
        '''
        Used for inspecting the state of FlexStoreManager
        '''
        logstr = f"{'='*30} FlexStoreManager {'='*30}\n"
        logstr += f"port: {self.port}\n"
        logstr += f"block_size: {self.block_size}\n"
        logstr += f"head_size: {self.head_size}\n"

        logstr += f"global_kv_cache_blocks:\n"
        kv_size = np.prod(self.get_key_block_shape())
        for k, v in self.global_free_blocks.items():
            logstr += f"   cuda:{k}: {len(v)} blocks, "
            logstr += f"{len(v) * 2 * 2 * kv_size / 1e9:.3f} GB\n"
        logstr += f"model_kv_cache_info:\n"
        for k, v in self.cache_view.items():
            logstr += f"   model {k}: (group size {v.group_size})\n"
            for dev, groups in v.avaliable_groups.items():
                logstr += f"       cuda:{dev}: "
                logstr += f"total {v.num_total_groups} groups, "
                logstr += f"free {len(groups)} groups, "
                logstr += f"allocated {v.num_total_groups - len(groups)} groups\n"
        logstr += f"{'='*50}\n"
        return logstr

    def memory_stats(self, prefix: Optional[str] = None):
        max_allocated_memory = torch.cuda.max_memory_allocated() / 1024**3
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        reserved_memory = torch.cuda.memory_reserved() / 1024**3
        cached_memory = torch.cuda.memory_cached() / 1024**3
        logger.info(f"{prefix} Memory Stats: "
                    f"Allocated {allocated_memory:.2f} GB, "
                    f"Max Allocated {max_allocated_memory:.2f} GB, "
                    f"Reserved {reserved_memory:.2f} GB, "
                    f"Cached {cached_memory:.2f} GB")


def get_gpu_memory(gpu: int = 0) -> int:
    """Returns the total memory of the GPU in bytes."""
    return torch.cuda.get_device_properties(gpu).total_memory


def get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def get_tensor_metadata(tensor: torch.Tensor) -> Dict:
    storage = tensor.storage()
    t = storage._share_cuda_()
    return {
        "tensor_size": tensor.size(),
        "tensor_stride": tensor.stride(),
        "tensor_offset": tensor.storage_offset(),
        "storage_cls": type(storage),
        "dtype": tensor.dtype,
        "storage_device": t[0],
        "storage_handle": t[1],
        "storage_size_bytes": t[2],
        "storage_offset_bytes": t[3],
        "requires_grad": tensor.requires_grad,
        "ref_counter_handle": t[4],
        "ref_counter_offset": t[5],
        "event_handle": t[6],
        "event_sync_required": t[7],
    }
