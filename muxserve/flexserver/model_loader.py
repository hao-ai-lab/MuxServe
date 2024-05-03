"""Utilities for selecting and loading models."""
import contextlib
from typing import Type, List, Optional, Dict

import torch
import torch.nn as nn
from torch.multiprocessing.reductions import rebuild_cuda_tensor
from transformers import PretrainedConfig

from vllm.config import ModelConfig
from vllm.model_executor.weight_utils import initialize_dummy_weights
from vllm.zmq_tool import ZMQClient

from muxserve.flexserver.models import *  # pylint: disable=wildcard-import
from muxserve.logger import get_logger

logger = get_logger()

_MODEL_REGISTRY = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "LLaMAForCausalLM": LlamaForCausalLM,  # For decapoda-research/llama-*
}


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")


def update_parameters(model: nn.Module, data: Dict[str, dict]):
    for param_name, param in model.named_parameters():
        cuda_tensor = rebuild_cuda_tensor(torch.Tensor, **(data[param_name]))
        assert param.shape == cuda_tensor.shape
        assert cuda_tensor.is_cuda
        param.data = cuda_tensor


def load_from_server(model: nn.Module, tcp_client: ZMQClient,
                     model_config: ModelConfig):
    # suppose our model was deployed on single card now
    logger.info('connecting server '
                f'from client cuda{str(torch.cuda.current_device())}')

    # ask for the server about the weight
    rank = torch.distributed.get_rank()
    tcp_client.send_pyobj(["weight", [rank, model_config.model_name]])

    logger.info('client: connected, waiting data')
    data = tcp_client.recv_pyobj()
    logger.info('client: data received, rebuilding and printing')

    update_parameters(model, data)
    # could be commented because of assert cuda_tensor.is_cuda
    model = model.cuda()


def get_model(model_config: ModelConfig,
              tcp_client: Optional[ZMQClient] = None,
              **kwargs) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config)

    with _set_default_torch_dtype(model_config.dtype):

        # Create a model instance.
        # The weights will be initialized as empty tensors.
        model = model_class(model_config.hf_config, **kwargs)

        if tcp_client is not None:
            load_from_server(model, tcp_client, model_config)
        else:
            if model_config.load_format == "dummy":
                model = model.cuda()
                # NOTE(woosuk): For accurate performance evaluation, we assign
                # random values to the weights.
                initialize_dummy_weights(model)
            else:
                # Load the weights from the cached or downloaded files.
                model.load_weights(model_config.model,
                                   model_config.download_dir,
                                   model_config.load_format,
                                   model_config.revision)
                model = model.cuda()

    return model.eval()
