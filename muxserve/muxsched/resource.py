import random
import numpy as np
from typing import Dict, List, Tuple

import torch
from transformers import AutoConfig
from muxserve.config import MuxServeConfig


def get_gpu_memory(gpu: int = 0) -> int:
    """Returns the total memory of the GPU in bytes."""
    return torch.cuda.get_device_properties(gpu).total_memory


class SMPart:

    def __init__(self, id):
        self.id = id

        self.ref_count = 0

    def acquire(self):
        self.ref_count += 1

    def release(self):
        self.ref_count -= 1

    def is_idle(self):
        return self.ref_count == 0

    def is_overloaded(self):
        return self.ref_count > 1


class SMResource:

    def __init__(self, overload_threshold: int = 4):
        self.overload_threshold = overload_threshold

        self._sm_parts: Dict[int, SMPart] = {}
        self.free_sms: List[SMPart] = []
        self.unoverloaded_sms: List[int] = []
        for i in range(10):
            self._sm_parts[i] = SMPart(i)
            self.free_sms.append(self._sm_parts[i])
            self.unoverloaded_sms.append(i)

    def can_allocate(self, num_sms: int, overload: bool = False) -> bool:
        extra = min(self.overload_threshold, len(
            self.unoverloaded_sms)) if overload else 0
        return len(self.free_sms) + extra >= num_sms

    def allocate(self, num_sms: int, overload: bool = False) -> List[SMPart]:
        ret = []
        for _ in range(num_sms):
            if len(self.free_sms) > 0:
                sm = self.free_sms.pop()
            else:
                sm_id = random.choice(self.unoverloaded_sms)
                sm = self._sm_parts[sm_id]
                self.unoverloaded_sms.remove(sm_id)
            sm.acquire()
            ret.append(sm)
        return ret

    def free(self, sms: List[SMPart]) -> None:
        for sm in sms:
            sm.release()
            if sm.is_idle():
                self.free_sms.append(sm)
            if not sm.is_overloaded() and sm.id not in self.unoverloaded_sms:
                self.unoverloaded_sms.append(sm.id)

    @property
    def num_free_sms(self):
        return len(self.free_sms)

    @property
    def num_overloaded_sms(self):
        return len(self._sm_parts) - len(self.unoverloaded_sms)


class CacheResource:

    def __init__(self, muxserve_config: MuxServeConfig):
        self.config = muxserve_config

        self.block_size = self.config.block_size
        self.head_size = 128
        self.num_blocks = self.get_num_gpu_blocks()

        # alloc info
        self.free_blocks = self.num_blocks
        self.request_blocks: Dict[str, Tuple[int, int]] = {}

        # job info
        self.model_blocks = {}
        for job_config in self.config.job_configs:
            model_config = AutoConfig.from_pretrained(job_config.model)
            # FIXME: single card currently
            assert job_config.tensor_parallel_size == 1, "tensor parallel is not supported"
            num_heads = model_config.num_attention_heads
            num_layers = model_config.num_hidden_layers
            self.model_blocks[model_config.model] = num_heads * num_layers

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

    def allocate(self, model: str, num_tokens: int):
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        num_blocks = num_blocks * self.model_blocks[model]

    def append_slot(self, model: str):
        pass
