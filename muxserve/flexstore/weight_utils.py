# Borrowed from vllm
import os
import glob
import json
import torch
import filelock
import numpy as np
from tqdm.auto import tqdm
from safetensors.torch import safe_open
from huggingface_hub import snapshot_download
from typing import Any, Iterator, List, Optional, Tuple


def initialize_dummy_weights(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3,
) -> None:
    """Initialize model weights with random values.

    The model weights must be randomly initialized for accurate performance
    measurements. Additionally, the model weights should not cause NaNs in the
    forward pass. We empirically found that initializing the weights with
    values between -1e-3 and 1e-3 works well for most models.
    """
    for param in model.state_dict().values():
        param.data.uniform_(low, high)


def convert_pyslice_to_tensor(x: Any) -> torch.Tensor:
    """convert PySafeSlice object from safetensors to torch.Tensor

    PySafeSlice object supports indexing, which is done before loading the
    actual tensor and can reduce the amount of memory being read into the
    memory. However, it does not support more advanced functionalities
    like `.view()` or `.t()`. Therefore, if we need to modify the loaded
    tensor with these more complicated operators, we need to convert to
    tensor first.
    """
    if not isinstance(x, torch.Tensor):
        x = x[:]
    return x


class Disabledtqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def get_lock(model_name_or_path: str, cache_dir: Optional[str] = None):
    lock_dir = cache_dir if cache_dir is not None else "/tmp"
    lock_file_name = model_name_or_path.replace("/", "-") + ".lock"
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name))
    return lock


def prepare_hf_model_weights(
    model_path: str,
    cache_dir: Optional[str] = None,
    use_safetensors: bool = False,
    fall_back_to_pt: bool = True, # safetensors找不到时用pt重试
    revision: Optional[str] = None,
) -> Tuple[str, List[str], bool]:
    """
    下载或获取本地模型权重，返回权重文件信息
    """
    is_local = os.path.isdir(model_path)
    if use_safetensors:
        allow_patterns = ["*.safetensors"]
    else:
        allow_patterns = ["*.bin", "*.pt"]
    if not is_local:
        # 下载模型，文件锁防止重复下载
        with get_lock(model_path, cache_dir):
            hf_folder = snapshot_download(
                model_path,
                allow_patterns=allow_patterns,
                cache_dir=cache_dir,
                tqdm_class=Disabledtqdm,
                revision=revision
            )
    else:
        hf_folder = model_path
    hf_weights_files: List[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
    if not use_safetensors:
        # 排除推理不必要的文件 https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
        blacklist = [
            "training_args.bin",
            "optimizer.bin",
            "optimizer.pt",
            "scheduler.pt",
            "scaler.pt",
        ]
        hf_weights_files = [
            f for f in hf_weights_files if not any(f.endswith(x) for x in blacklist)
        ]
    if len(hf_weights_files) == 0:
        if use_safetensors and fall_back_to_pt:
            return prepare_hf_model_weights(
                model_path,
                cache_dir=cache_dir,
                use_safetensors=False,
                fall_back_to_pt=False,
                revision=revision
            )
        else: raise RuntimeError(f"Cannot find any model weights with `{model_path}`")
    return hf_folder, hf_weights_files, use_safetensors


def hf_model_weights_iterator(
    model_path: str,
    cache_dir: Optional[str] = None,
    load_format: str = "auto",
    revision: Optional[str] = None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """
    逐个读取并返回模型权重
    """
    use_safetensors = False
    use_np_cache = False
    fall_back_to_pt = False
    if load_format == "auto":
        use_safetensors = True
        fall_back_to_pt = True
    elif load_format == "safetensors":
        use_safetensors = True
    elif load_format == "npcache":
        use_np_cache = True
    elif load_format == "pt":
        pass
    else:
        raise ValueError(f"Unknown load_format: {load_format}")
    hf_folder, hf_weights_files, use_safetensors = prepare_hf_model_weights(
        model_path,
        cache_dir=cache_dir,
        use_safetensors=use_safetensors,
        fall_back_to_pt=fall_back_to_pt,
        revision=revision
    )
    if use_np_cache:
        assert use_safetensors is False
        np_folder = os.path.join(hf_folder, "np")
        os.makedirs(np_folder, exist_ok=True)
        weight_names_file = os.path.join(np_folder, "weight_names.json")
        weight_names = []
        if not os.path.exists(weight_names_file):
            with get_lock(model_path, cache_dir):
                for bin_file in hf_weights_files: # 仅支持.bin文件
                    state = torch.load(bin_file, map_location="cpu")
                    for name, param in state.items():
                        param_path = os.path.join(np_folder, name)
                        with open(param_path, "wb") as f:
                            np.save(f, param.cpu().detach().numpy()) # 转化换numpy加速导入
                        weight_names.append(name)
                    del state
                with open(weight_names_file, "w") as f:
                    json.dump(weight_names, f)
        else:
            with open(weight_names_file, "r") as f:
                weight_names = json.load(f)
        for name in weight_names:
            param_path = os.path.join(np_folder, name)
            with open(param_path, "rb") as f:
                param = np.load(f)
            yield name, torch.from_numpy(param)
    elif use_safetensors:
        for st_file in hf_weights_files:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():
                    param = f.get_slice(name)
                    yield name, convert_pyslice_to_tensor(param)
    else:
        for bin_file in hf_weights_files:
            state = torch.load(bin_file, map_location="cpu")
            for name, param in state.items():
                yield name, param
            del state
    torch.cuda.empty_cache()