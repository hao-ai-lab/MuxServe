"""A GPU worker class."""
import os
import numpy as np
import torch
import torch.distributed

from typing import Dict, List, Tuple, Optional, Union

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.model_executor import InputMetadata
from vllm.sequence import SamplerOutput

from vllm.worker.worker import Worker

from muxserve.logger import get_logger
from muxserve.flexserver.model_loader import get_model

logger = get_logger()

KVCache = Tuple[torch.Tensor, torch.Tensor]


class PipeWorker(Worker):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        rank: Optional[int] = None,
        distributed_init_method: Optional[str] = None,
        flexstore_port: str = None,
        stage_id: int = 0,
        model_id: int = 0,
    ) -> None:
        super().__init__(model_config, parallel_config, scheduler_config, rank,
                         distributed_init_method, flexstore_port)

        self.stage_id = stage_id
        self.model_id = model_id
        self.num_stages = self.parallel_config.pipeline_parallel_size

    def init_model(self):
        total_num_layers = self.model_config.hf_config.num_hidden_layers
        partition = self.pipeline_split(total_num_layers, self.num_stages)
        self.num_hidden_layers = partition[self.stage_id]
        if self.stage_id == 0:
            logger.info(
                f"Model id: {self.model_id}, Name: {self.model_config.model}, "
                f"Total layers: {total_num_layers}, "
                f"Pipeline partition (size {self.num_stages}): {partition}")

        pre_process = (self.stage_id == 0)
        post_process = (self.stage_id == self.num_stages - 1)
        self.model = get_model(self.model_config,
                               self.tcp_client,
                               num_hidden_layers=self.num_hidden_layers,
                               pre_process=pre_process,
                               post_process=post_process)

    @classmethod
    def pipeline_split(cls, num_hidden_layers: int,
                       pipeline_parallel_size: int):
        num_layer_per_part = num_hidden_layers // pipeline_parallel_size
        partition = [num_layer_per_part] * pipeline_parallel_size
        if num_hidden_layers % pipeline_parallel_size == pipeline_parallel_size - 1:
            start_partition = 0
        else:
            start_partition = 1
        for i in range(num_hidden_layers % pipeline_parallel_size):
            partition[start_partition + i] += 1
        return partition

    def _prepare_inputs_metadata(self, prompt_lens: List[int],
                                 slot_mapping: torch.Tensor,
                                 context_lens: torch.Tensor,
                                 max_context_len: int,
                                 block_tables: torch.Tensor) -> InputMetadata:

        input_metadata = InputMetadata(
            seq_groups=None,
            seq_data=None,
            prompt_lens=prompt_lens,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            max_context_len=max_context_len,
            block_tables=block_tables,
            sliding_window=None,
        )
        return input_metadata

    @torch.inference_mode()
    def pure_forward(
        self,
        input_tensor: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> Union[torch.Tensor, List[int]]:
        # Execute the model.
        output = self.model(input_ids=input_tensor,
                            positions=positions,
                            kv_caches=self.gpu_cache,
                            input_metadata=input_metadata,
                            cache_events=None)
        return output
