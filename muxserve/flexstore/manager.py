import os
import re
import copy
import time
import torch
import numpy as np
from transformers import AutoConfig, PretrainedConfig
from typing import Dict, Iterator, List, Optional, Tuple
from muxserve.logger import get_logger
from muxserve.zmq_utils import ZMQServer
from muxserve.memory_manager import KVStorage
from muxserve.config import JobConfig, MuxServeConfig
from muxserve.flexserver.pipeworker import PipeWorker
from muxserve.flexstore.weight_utils import hf_model_weights_iterator

logger = get_logger()

KVCache = Tuple[torch.Tensor, torch.Tensor]

def replace_numbers_with_value(s: str, replacement_value: int):
    """ 将所有数字替换为指定值 """
    regex_pattern = r'\d+'
    result = re.sub(regex_pattern, str(replacement_value), s)
    return result


def grasp_num(s: str):
    """ 提取字符串首个数字并转换为int """
    regex_pattern = r'\d+'
    result = re.findall(regex_pattern, s)
    return int(result[0])


class WeightStorage:
    """ 管理模型权重 """
    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        job_config: JobConfig,
        model_config: PretrainedConfig,
        rank_start: int,
        rank_end: int
    ) -> None:
        placement = job_config.placement[0] # 放置在哪些GPU上
        self.dtype = job_config.model_dtype # 如torch.float16
        logger.info(f"load_weight: {job_config.model_path}, placement: {placement}, dtype: {self.dtype}")
        """记录GPU信息：{GPU id: {权重名称: 权重的参数}}"""
        self.data: Dict[int, Dict[str, torch.Tensor]] = {k: {} for k in placement}
        self.metadata: Dict[int, Dict[str, Dict]] = {k: {} for k in placement}
        """投影矩阵拼接，分片"""
        reshaped_weights = WeightStorage.reshape_weights(weights, job_config, model_config)
        """计算流水线负载分配"""
        # tp_size * pp_size = placement.size
        tp_size = job_config.tensor_parallel_size # 层内的并行性，单个张量用tp个GPU计算
        pp_size = job_config.pipeline_parallel_size # 层间的并行性，不同层按pipeline由不同GPU（组）计算
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
            # 输入<输出，对投影输出分片，结果需要拼接
            col_split = ["qkv_proj", "gate_up_proj", "embed_tokens", "lm_head"]
            # 输入>输出，对投影输入分片，结果需要求和
            row_split = ["down_proj", "o_proj"]
            # for pipeline parallel
            pre_process_weights = ["model.embed_tokens.weight"] # 嵌入层
            post_process_weights = ["model.norm.weight", "lm_head.weight"] # 归一输出层
            # 每个pipeline阶段处理的层数
            pipeline_partition: List[int] = PipeWorker.pipeline_split(model_config.num_hidden_layers, pp_size)
        else:
            raise RuntimeError("Only Llama supported now")

        logger.info(f"### Begin to place weights on GPUs ...")
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for name, val in reshaped_weights.items():
            is_column_parallel = False
            for p in col_split:
                if p in name:
                    shard_size = val.shape[0] // tp_size # Y = WX，对W按行分片，Y = [ W1X | ... | WiX ]^T
                    for idx, dev_idx in enumerate(placement):
                        if dev_idx < rank_start or dev_idx >= rank_end: continue
                        local_rank = dev_idx - rank_start
                        # 计算并行化编号
                        pp_rank = idx // tp_size
                        tp_rank = idx % tp_size
                        if name in pre_process_weights:
                            # embed_tokens层必须在pipeline首段
                            mapped_name = name
                            if pp_rank != 0: continue
                        elif name in post_process_weights:
                            # 输出层必须在pipeline尾段
                            mapped_name = name
                            if pp_rank != pp_size - 1: continue
                        else:
                            # 隐藏层必须在对应pipeline阶段
                            original_layer_idx = grasp_num(name)
                            placed_stage = -1
                            while original_layer_idx >= 0:
                                mapped_layer_idx = original_layer_idx
                                placed_stage += 1
                                original_layer_idx -= pipeline_partition[placed_stage]
                            if pp_rank != placed_stage: continue
                            # 名称改为阶段内部编号
                            mapped_name = replace_numbers_with_value(name, mapped_layer_idx)
                        # 获取第一维分片
                        weight = val[tp_rank * shard_size: (tp_rank + 1) * shard_size]
                        self.data[dev_idx][mapped_name] = weight.to(f"cuda:{local_rank}", dtype=self.dtype)
                    is_column_parallel = True
                    break
            if is_column_parallel:
                continue

            is_row_parallel = False
            for p in row_split:
                if p in name:
                    shard_size = val.shape[1] // tp_size # Y = WX，对W按列分片，对X按行分片，Y = sum(WiXi)
                    for idx, dev_idx in enumerate(placement):
                        if dev_idx < rank_start or dev_idx >= rank_end: continue
                        local_rank = dev_idx - rank_start
                        # 计算并行化编号
                        pp_rank = idx // tp_size
                        tp_rank = idx % tp_size
                        # 隐藏层必须在对应pipeline阶段
                        original_layer_idx = grasp_num(name)
                        placed_stage = -1
                        while original_layer_idx >= 0:
                            mapped_layer_idx = original_layer_idx
                            placed_stage += 1
                            original_layer_idx -= pipeline_partition[placed_stage]
                        if pp_rank != placed_stage: continue
                        # 名称改为阶段内部编号
                        mapped_name = replace_numbers_with_value(name, mapped_layer_idx)
                        # 获取第二维分片
                        weight = val[:, tp_rank * shard_size: (tp_rank + 1) * shard_size]
                        self.data[dev_idx][mapped_name] = weight.to(f"cuda:{local_rank}", dtype=self.dtype)
                    is_row_parallel = True
                    break
            if is_row_parallel:
                continue

            # 其它层（归一化层）每个模型一份权重
            for idx, dev_idx in enumerate(placement):
                if dev_idx < rank_start or dev_idx >= rank_end: continue
                local_rank = dev_idx - rank_start
                pp_rank = idx // tp_size
                if name in post_process_weights and pp_rank != pp_size - 1: continue
                self.data[dev_idx][name] = val.to(f"cuda:{local_rank}", dtype=self.dtype)
                
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        logger.info(f"### Cost of data transfer (CPU -> GPU): {end_time-start_time:.3f} s")
        for dev_idx, weight_info in self.data.items():
            for weight_name, weight_val in weight_info.items():
                self.metadata[dev_idx][weight_name] = get_tensor_metadata(weight_val)

    def __str__(self) -> str:
        res = "{\n"
        for dev_idx, weight_info in self.data.items():
            for weight_name, weight_val in weight_info.items():
                res += f"  {weight_name}_rank_{dev_idx}: {weight_val.shape}; {weight_val.device}\n"
        return res + "}\n"

    @classmethod
    def from_iter(
        cls,
        iter: Iterator[Tuple[str, torch.Tensor]],
        job_config: JobConfig,
        model_config: PretrainedConfig,
        rank_start: int,
        rank_end: int
    ):
        weights = {layer_name: data for (layer_name, data) in iter}
        return cls(weights, job_config, model_config, rank_start, rank_end)

    @staticmethod
    def reshape_weights(weights: Dict[str, torch.Tensor], job_config: JobConfig, model_config: PretrainedConfig):
        '''
        拼接[w_q, w_k, w_v]和[w_gate, w_up]
        结果weights[name]中指定权重在指定GPU上的分片，起始位置为tp_offset + proj_offset
        '''
        tp_size = job_config.tensor_parallel_size
        weights_copy = copy.deepcopy(weights)

        if model_config.model_type == "llama":
            """Llama使用GQA，Q的切片和KV的切片大小不同"""
            q_proj_shard_size = (model_config.hidden_size // tp_size) # 单个GPU处理的维度
            # tp较大时，每个KV头在多个GPU上计算，需要复制KV头
            num_kv_heads_replicas = max(1, tp_size // model_config.num_key_value_heads)
            # tp较小时，每个GPU上分为多个KV头，KV头需要共享GPU
            num_kv_heads_per_gpu = max(1, model_config.num_key_value_heads // tp_size)
            # GQA中，合并KV的大小为(hidden_size, head_dim * KV_num)，其中head_dim = hidden_size // Q_head_num
            # 因此张量并行时，每个GPU处理一部分KV头，分到的KV权重为(hidden_size, head_dim * KV_num_per_GPU)
            kv_proj_shard_size = (model_config.hidden_size // model_config.num_attention_heads * num_kv_heads_per_gpu)
            # 将QKV权重拼接成大矩阵，offset为在大矩阵中的偏移量，[ Q | K | V ] = X [ W_Q | W_K | W_V ]
            # 单个GPU只用关注部分[ Q | K | V ]，因此使用shard_size计算部分输出
            attn_weight_specs = [
                # (weight_name, shard_size, offset)
                ("q_proj", q_proj_shard_size, 0),
                ("k_proj", kv_proj_shard_size, q_proj_shard_size),
                ("v_proj", kv_proj_shard_size, q_proj_shard_size + kv_proj_shard_size),
            ]
            per_rank_qkv_proj_size = q_proj_shard_size + kv_proj_shard_size * 2 # 单个GPU Attention总输出维度
            """        
            Llama的MLP层结构为down_proj(gate(X) * swish(up_proj(X)))
            gate：门控投影，[hidden_size, intermediate_size]
            up_proj：上采样投影，[hidden_size, intermediate_size]
            swish：激活函数
            down_proj：下采样投影，[intermediate_size, hidden_size]
            """
            gate_up_shard_size = model_config.intermediate_size // tp_size # 单个GPU MLP中间层维度
            per_rank_gate_up_proj_size = gate_up_shard_size * 2 # 单个GPU MLP投影矩阵总维度\
            for name, loaded_weight in weights.items():
                # 旋转位置编码，不需要依赖权重文件加载
                if "rotary_emb.inv_freq" in name:
                    del weights_copy[name]
                    continue
                # q_proj: [q_num_heads*head_dim, hidden], k/v_proj: [k/v_num_heads*head_dim, hidden]
                # cat qkv along the 1st dim: [(2*kv_num_heads+q_num_heads)*head_dim, hidden]
                is_attn_weight = False
                for weight_name, shard_size, offset in attn_weight_specs:
                    if weight_name not in name: # 权重是否为q_proj，k_proj，v_proj之一
                        continue
                    cat_weight_name = name.replace(weight_name, "qkv_proj")
                    # 1. 初始化[ q_proj | k_proj | v_proj ]
                    if cat_weight_name not in weights_copy:
                        weights_copy[cat_weight_name] = torch.empty(
                            (per_rank_qkv_proj_size * tp_size, model_config.hidden_size),
                            dtype=torch.float16
                        )
                    # 2. 权重复制
                    cat_weight = weights_copy[cat_weight_name]
                    for tp_rank in range(tp_size):
                        if weight_name in ["k_proj", "v_proj"]:
                            shard_id = tp_rank // num_kv_heads_replicas # 可能多个GPU共用KV头
                        else:
                            shard_id = tp_rank
                        # 根据tp偏移加q/k/v偏移复制到相应位置
                        tp_offset = tp_rank * per_rank_qkv_proj_size
                        cat_weight[tp_offset + offset: tp_offset + offset + shard_size].copy_(
                            loaded_weight[shard_id * shard_size: (shard_id + 1) * shard_size]
                        ) # q/k/v矩阵在该GPU上的切片
                    del weights_copy[name]
                    is_attn_weight = True
                    break
                if is_attn_weight:
                    continue

                # gate_proj: [intermediate_size, hidden], up_proj: [intermediate_size, hidden]
                # cat `gate_proj` and `up_proj` along the 1st dim
                is_gate_up_weight = False
                for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                    if weight_name not in name:
                        continue
                    cat_weight_name = name.replace(weight_name, "gate_up_proj")
                    # 1. 初始化[ gate_proj | up_proj ]
                    if cat_weight_name not in weights_copy:
                        weights_copy[cat_weight_name] = torch.empty(
                            (loaded_weight.shape[0] * 2, loaded_weight.shape[1]),
                            dtype=torch.float16
                        )
                    # 2. 权重复制
                    cat_weight = weights_copy[cat_weight_name]
                    shard_size = gate_up_shard_size
                    for tp_rank in range(tp_size):
                        # 根据tp偏移加stride_id偏移复制到相应位置
                        tp_offset = tp_rank * per_rank_gate_up_proj_size
                        cat_weight[tp_offset + stride_id * shard_size: tp_offset + (stride_id + 1) * shard_size].copy_(
                            loaded_weight[tp_rank * shard_size: (tp_rank + 1) * shard_size]
                        ) # 投影矩阵在该GPU上的切片
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
        self.port = muxserve_config.flexstore_port
        self.head_size = muxserve_config.head_size
        self.block_size = muxserve_config.block_size
        # 集群分布式配置，查询节点信息
        use_openmpi = os.environ.get("OMPI_COMM_WORLD_SIZE", None) is not None
        use_mpich = os.environ.get("PMI_SIZE", None) is not None
        if use_openmpi:
            local_world_size = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', 1))
            rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
            world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
        elif use_mpich:
            local_world_size = int(os.environ.get('MPI_LOCALNRANKS', 1))
            rank = int(os.environ.get('PMI_RANK', 0))
            world_size = int(os.environ.get('PMI_SIZE', 1))
        else:
            local_world_size = muxserve_config.nproc_per_node
            rank = muxserve_config.node_rank
            world_size = muxserve_config.nnodes
        self.local_world_size = local_world_size # 单节点GPU数
        self.world_size = world_size * self.local_world_size # 总GPU数
        self.rank = rank # 节点编号
        # 节点GPU编号范围
        self.rank_start = rank * local_world_size
        self.rank_end = (rank + 1) * local_world_size
        devices: set[int] = set() # 所有GPU
        self.rank_to_dev: Dict[str, List[int]] = {} # {模型名称: [GPU列表]}，列表索引为rank，值为id
        for job in muxserve_config.job_configs:
            self.rank_to_dev[job.name] = []
            for dev_id in job.placement[0]:
                devices.add(dev_id)
                self.rank_to_dev[job.name].append(dev_id)
        self.devices = list(devices)
        logger.info(f"rank_to_dev: {self.rank_to_dev}")
        # 加载模型权重
        self.models_weights: Dict[str, WeightStorage] = self.load_models()
        self.memory_stats("After loading models")
        # 将所用GPU剩余内存初始化为KV-Cache块
        self.gpu_cache: Dict[int, KVCache] = self.allocate_gpu_cache()
        self.gpu_cache_matedata: Dict[int, Dict] = { # {dev_id: (k_cache_metadata, v_cache_metadata)}
            k: (get_tensor_metadata(v[0]), get_tensor_metadata(v[1])) for (k, v) in self.gpu_cache.items()
        }
        # block manager
        num_total_blocks = self.get_num_gpu_blocks()
        self.block_manager = KVStorage(num_total_blocks)
        self.model_cache_info: Dict[str, Tuple[int, int]] = {} # 单GPU模型权重：{model_name: (num_hidden, num_head)}
        self.model_occupy_info: Dict[str, int] = {} # {model_name: num_block_occupied}
        for job_config in self.config.job_configs:
            model_name = job_config.name
            model_config = AutoConfig.from_pretrained(job_config.model_path)
            # 同一层在每个GPU上的头数
            num_heads = model_config.num_attention_heads // job_config.tensor_parallel_size
            pipeline_partition = PipeWorker.pipeline_split(
                model_config.num_hidden_layers,
                job_config.pipeline_parallel_size
            )
            num_hidden_layers = max(pipeline_partition)
            self.model_cache_info[model_name] = (num_hidden_layers, num_heads)
            self.model_occupy_info[model_name] = 0
        self.memory_stats("After init cache view")
        # Store block_table from clients
        self.block_table_storage: Dict = {}
        # record blocks allocated for each request
        self.request_to_blocks: Dict[int, List[int]] = {}

    def load_models(self) -> Dict[str, WeightStorage]:
        """加载所有模型权重"""
        result: Dict[str, WeightStorage] = {}
        job_configs = self.config.job_configs
        for job_config in job_configs:
            tp_size = job_config.tensor_parallel_size
            pp_size = job_config.pipeline_parallel_size
            assert tp_size * pp_size <= self.world_size
            assert tp_size * pp_size == len(job_config.placement[0])
            logger.info(f"Rank {self.rank} starts loading {job_config.name} ({job_config.model_path}) ...")
            weight_iter = hf_model_weights_iterator(job_config.model_path)
            model_config = AutoConfig.from_pretrained(job_config.model_path)
            result[job_config.name] = WeightStorage.from_iter(weight_iter, job_config, model_config, self.rank_start, self.rank_end)
        return result

    def get_num_gpu_blocks(self) -> int:
        """获取单个GPU可分配的总块数，需为128倍数"""
        total_mem_each_gpu = get_gpu_memory()
        avaliable_mem_each_gpu = self.config.gpu_memory_utilization * total_mem_each_gpu
        avaliable_mem_each_gpu = round(avaliable_mem_each_gpu) - 1
        # block_mem = dtype_size * (k_block_shape + v_block_shape)
        block_mem = 2 * (np.prod(self.get_key_block_shape()) + np.prod(self.get_value_block_shape()))
        max_num_blocks = (avaliable_mem_each_gpu // block_mem // 128) * 128 # 块总数为128倍数
        return max_num_blocks

    def get_key_block_shape(self, element_size=2) -> Tuple[int, int, int]:
        """Return (split_num, block_size, split_size)"""
        split_size = 16 // element_size
        split_num = self.head_size // split_size
        # K-Cache对seq_len维度分块储存，每块block_size
        # Q * K_block^T 为 [1, head_size] * [head_size, block_size] = [1, block_size]
        # 运算输入>输出，对行分片，对分片乘法结果求和
        # 行分片物理大小为16B，元素数量为split_size
        return (
            split_num,
            self.block_size,
            split_size,
        )

    def get_value_block_shape(self) -> Tuple[int, int]:
        """Return (head_size, block_size)"""
        # o * V 为 [1 * block_size] * [block_size, head_size] = [1 * head_size]
        # 不需要分片操作
        return (
            self.head_size,
            self.block_size,
        )

    def allocate_gpu_cache(self) -> Dict[int, KVCache]:
        '''Return {dev_id: (k_cache_block, v_cache_block)}'''
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
                dtype=torch.float16,
                device=f"cuda:{local_rank}"
            )
            v_cache = torch.empty(
                size=(num_gpu_blocks, *v_block_shape),
                dtype=torch.float16,
                device=f"cuda:{local_rank}"
            )
            gpu_cache[device_id] = (k_cache, v_cache)
            logger.info(
                f"Allocate {num_gpu_blocks} blocks \
                ({2*k_cache.nelement()*2/1e9} GB) KV Cache on cuda:{local_rank}"
            )
        return gpu_cache

    @staticmethod
    def parse_request(req: Tuple):
        """过滤请求"""
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
            "free_cache",
            "lock_init",
            "log_stats",
            "exit"
        ]:
            return None
        return (req_type, req_args)

    def deploy(self):
        '''
        分布式框架核心循环

        Args:
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

            free_cache request format:
                Tuple["free_cache", [{rank}, {model_name}, {layer-wise-block_table}]]
                - memory_manager will collect cache blocks freed by this request
        '''
        tcp_server = ZMQServer("localhost", self.port)
        proc_id = 1
        proc_id_map = {}
        lock_tensor = torch.tensor(0, dtype=torch.int, device='cuda:0')
        ready_processes = 0
        process_in_warmup = False

        logger.info(f"Memory manager is listening on {self.port} ...")
        # 服务器主循环
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
            elif req_type == "init_finished": # 检查所有进程是否初始化完成
                ret = ready_processes == self.config.num_runtime_processes
            elif req_type == "query_num_ready_processes":
                ret = ready_processes
            elif req_type == "get_num_blocks": # 获取单个GPU可分配的块数
                ret = self.get_num_gpu_blocks()
            elif req_type == "start_warmup":
                if process_in_warmup:
                    ret = False
                else:
                    process_in_warmup = True
                    ret = True
            elif req_type == "warmup_ready": # 通知进程初始化完成
                process_in_warmup = False
                ready_processes += 1
                ret = ready_processes == self.config.num_runtime_processes
                logger.info(f"{ready_processes}/{self.config.num_runtime_processes} processes ready")
            elif req_type == "weight": # 获取指定GPU上指定模型的权重参数
                logger.info(f"Receive {req_type}, {req_args}")
                rank, model_name = req_args
                req_weight = self.models_weights[model_name]
                dev_id = self.rank_to_dev[model_name][rank]
                ret = req_weight.metadata[dev_id]
            elif req_type == "cache_init": # 获取获取指定GPU上指定模型的缓存参数
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
                ret = self.block_manager.allocate_batch(batch_info, num_layers, num_heads)
                logger.info(f"cache_alloc: {ret}, {type(ret)} size: {ret.shape}")
            elif req_type == "free_cache":
                model_name, finished_request_ids = req_args
                num_freed_blocks = self.block_manager.free_batch(finished_request_ids)
                self.model_occupy_info[model_name] -= num_freed_blocks
                ret = None
                logger.info(f"free_cache: request: {finished_request_ids}, block_num: {num_freed_blocks}")
            elif req_type == "log_stats":
                logger.info("Receive log_stats")
                ret = None
            elif req_type == "exit":
                logger.info(f"Receive {req_type}, exit...")
                del self.models_weights
                break
            else:
                logger.warning(f"Receive {req_type}, {req_args}")
            tcp_server.send_pyobj(ret)

    def log_cache_usage(self):
        """打印内存块占用状态"""
        logstr = "[Block Usage] "
        for model_name, num_blocks in self.model_occupy_info.items():
            logstr += f"{model_name}: {num_blocks} ,"
        logstr = logstr[:-2]
        logger.info(logstr)

    def memory_stats(self, prefix: Optional[str] = None):
        """打印内存分配状态"""
        max_allocated_memory = torch.cuda.max_memory_allocated() / 1024**3 # 历史峰值
        allocated_memory = torch.cuda.memory_allocated() / 1024**3 # 当前活跃
        reserved_memory = torch.cuda.memory_reserved() / 1024**3 # pytorch已申请
        logger.info(
            f"{prefix} Memory Stats: \n"
            f"Allocated {allocated_memory:.2f} GB, "
            f"Max Allocated {max_allocated_memory:.2f} GB, "
            f"Reserved {reserved_memory:.2f} GB"
        )


def get_gpu_memory(gpu: int = 0) -> int:
    """ 获取指定GPU物理总内存大小 """
    return torch.cuda.get_device_properties(gpu).total_memory


def get_tensor_metadata(tensor: torch.Tensor) -> Dict:
    """ 获取张量参数 """
    storage = tensor.storage()
    # storage = tensor.untyped_storage()
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
        "ref_counter_handle": t[4],
        "ref_counter_offset": t[5],
        "event_handle": t[6],
        "event_sync_required": t[7],
        "requires_grad": tensor.requires_grad,
    }