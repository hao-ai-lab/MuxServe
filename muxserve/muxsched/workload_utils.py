"""Workload definition
Borrowed from https://github.com/alpa-projects/mms/blob/main/alpa_serve/simulator/workload.py
"""
import os
import json
import yaml
import pickle
import random
import argparse
import numpy as np
import dataclasses
from copy import deepcopy
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_WARMUP = 10
DEFAULT_DATASET_PATH = "/data/lrc/workspace/dataset/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json"
DEFAULT_TOKENIZER_PATH = "meta-llama/Llama-3.2-3B"
DEFAULT_TOKENIZED_CACHE = "/data/lrc/workspace/dataset/ShareGPT_V3_llama_tokenized.cache"
eps = 1e-6

model_rank_mapping = {
    "llm-0": "dummy-lora-7b-rank-8-0",
    "llm-1": "dummy-lora-7b-rank-8-10",
    "llm-2": "dummy-lora-7b-rank-8-2",
    "llm-3": "dummy-lora-7b-rank-8-5"
}

def to_str_round(x: Any, decimal: int = 6):
    """ Print a python object but round all floating point numbers. """
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple, np.ndarray)):
        tmp_str = ", ".join([to_str_round(y, decimal=decimal) for y in x])
        return "[" + tmp_str + "]"
    if isinstance(x, dict):
        return str({k: to_str_round(v, decimal=decimal) for k, v in x.items()})
    if isinstance(x, (int, np.int32, np.int64)):
        return str(x)
    if isinstance(x, (float, np.float32, np.float64)):
        format_str = f"%.{decimal}f"
        return format_str % x
    if x is None:
        return str(x)
    raise ValueError("Invalid value: " + str(x))


@dataclasses.dataclass
class Request:
    """ 用于跟踪单一请求的全周期 """
    model_name: str
    slo: Optional[float]
    idx: int
    time_stamp: Dict  # debug only
    data: Any
    submit_time: float = None  # This will be filled later
    prefill_end_time: float = None  # This will be filled later
    decode_submit_time: float = None  # This will be filled later
    end_time: float = None  # This will be filled later
    is_prefill: bool = True
    output: str = None
    output_idx: int = 0
    output_tokens: Optional[List[int]] = None

    # FIXME: ad-hoc, for test
    def __lt__(self, other):
        return self.idx < other.idx

    def __hash__(self):
        return hash(self.idx)

# 对每个模型维护统计结果
PerModelStatsResult = namedtuple(
    "PerModelStatsResult", (
        "name", "num_requests", "goodput", "throughput",
        "latency_mean", "latency_std",
        "latency_p90", "latency_p95", "latency_p99", "latency",
        "request_starts", "request_finishes"
    )
)
# 对每个GPU维护统计结果
PerDeviceStatsResult = namedtuple("PerDeviceStatsResult", ("num_requests"))


@dataclasses.dataclass
class StatsResult:
    """ 整体统计结果 """
    per_model_stats: List[PerModelStatsResult]
    group_num_requests: List[int]
    goodput: float
    latency_mean: float
    num_requests: int
    request_rate: float


class ArrivalProcess(ABC): # 接口

    @abstractmethod
    def rate(self):
        """Return the mean arrival rate."""
        raise NotImplementedError()

    @abstractmethod
    def cv(self):
        """Return the coefficient of variation of the gap between the requests."""
        raise NotImplementedError()

    @abstractmethod
    def generate_arrivals(self, start: float, duration: float, seed: int = 0):
        raise NotImplementedError()

    @abstractmethod
    def generate_workload(
        self,
        model_name: str,
        start: float,
        duration: float,
        slo: Optional[float] = None,
        seed: int = 0
    ):
        """
        Generate a workload with the arrival process.

        Args:
            model_name (str): Name of the model.
            start (float): The start time of the workload.
            duration (float): The duration of the workload.
            slo (Optional[float]): The service level objective of each model.
            seed (int): The random seed.
        """
        raise NotImplementedError()

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"rate={self.rate()}, "
            f"cv={self.cv()})"
        )

    def params(self):
        return self.rate(), self.cv()


class DeterministicProcess(ArrivalProcess):
    """ Deterministic arrival process. """

    def __init__(self, arrival_rate: float):
        """
        Create a deterministic arrival process.

        Args:
            arrival_rate (float): The arrival rate of the process.
            The gap between the requests is 1 / arrival_rate seconds.
        """
        self.rate_ = arrival_rate

    def rate(self):
        return self.rate_

    def cv(self):
        return 0

    def generate_arrivals(
        self,
        start: float,
        duration: float,
        num_requests: Optional[int] = None,
        seed: int = 0
    ):
        pass

    def generate_workload(
        self,
        model_name: str,
        start: float,
        duration: float,
        num_requests: Optional[int] = None,
        slo: Optional[float] = None,
        seed: int = 0
    ):
        """生成间隔固定的Workload，有次数上限或持续时间限制"""
        if num_requests is None:
            n_requests = max(int(duration * self.rate_), 1)
        else:
            n_requests = num_requests
        interval = 1 / self.rate_
        ticks = [start + i * interval for i in range(n_requests)]
        return Workload(ticks, [Request(model_name, slo, i, {}, None) for i in range(n_requests)])
                    

class TraceProcess():
    """ 从已有trace获取 """
    def __init__(self, trace=None):
        with open(trace, "r", encoding="utf-8") as f:
            self.trace = json.load(f)
        
    def generate_workload(
        self,
        model_name: str,
        tokenizer_path: str,
        slo: Optional[float] = None,
        workload_scale: float = 1,
        rate_scale: int = 1,
    ):
        requests = self.trace[1]
        tokenizer = get_tokenizer(tokenizer_path)
        ticks = []
        reqs = []
        i = 0
        token = 12000
        for req in requests:
            if req["adapter_dir"] == model_rank_mapping[model_name]:
                input_len = min(int(req["prompt_len"] * workload_scale), 2048)
                output_len = min(int(req["output_len"] * workload_scale), 2048)
                if input_len == 2048 or output_len == 2048:
                    print("Warning: Sequence length exceeds limit, truncated")
                sentence = ["Hello " * input_len]
                tokens = tokenizer(sentence).input_ids[0]
                tokens = [token] * len(tokens)
                data = (tokens, input_len, output_len)
                for _ in range(rate_scale):
                    ticks.append(req["req_time"])
                    reqs.append(Request(model_name, slo, i, {}, data))
                    i += 1
        return Workload(ticks, reqs)


class GammaProcess(ArrivalProcess):
    """ Gamma arrival process. """

    def __init__(self, arrival_rate: float, cv: float):
        """
        Initialize a gamma arrival process.

        Args:
            arrival_rate: mean arrival rate.
            cv: coefficient of variation. When cv == 1, the arrival process is
                Poisson process.
        """
        self.rate_ = arrival_rate
        self.cv_ = cv
        self.shape = 1 / (cv * cv)
        self.scale = cv * cv / arrival_rate

    def rate(self):
        return self.rate_

    def cv(self):
        return self.cv_

    def generate_arrivals(
        self,
        start: float,
        duration: float,
        num_requests: Optional[int] = None,
        seed: int = 0
    ):
        """gamma分布生成请求到达时刻"""
        np.random.seed(seed)
        if num_requests is None:
            batch_size = max(int(self.rate_ * duration * 1.2), 1)
        else:
            batch_size = num_requests
        # gamma分布生成间隔
        intervals = np.random.gamma(self.shape, self.scale, size=batch_size)
        pt = 0
        ticks = []
        cur = start + intervals[0]
        end = start + duration
        while cur < end:
            ticks.append(cur)
            pt += 1
            if pt >= batch_size:
                # 达到请求上限则停止生成
                if num_requests is not None:
                    break
                # 无请求上限则生成新一批请求
                intervals = np.random.gamma(self.shape,
                                            self.scale,
                                            size=batch_size)
                pt = 0
            cur += intervals[pt]
        return ticks

    def generate_workload(
        self,
        model_name: str,
        start: float,
        duration: float,
        num_requests: Optional[int] = None,
        slo: Optional[float] = None,
        seed: int = 0
    ):
        """生成间隔为gamma分布的Workload，有次数上限或持续时间限制"""
        ticks = self.generate_arrivals(start, duration, num_requests, seed)
        return Workload(
            ticks,
            [Request(model_name, slo, i, {}, None) for i in range(len(ticks))])


class PoissonProcess(GammaProcess):
    """ Poisson arrival process. """

    def __init__(self, arrival_rate: float):
        """
        Initialize a Poisson arrival process.
        协方差=1时，伽马分布退化为泊松分布
        
        Args:
            arrival_rate: The mean arrival rate.
        """
        super().__init__(arrival_rate, 1)


class Workload:
    """ 顺序排列的请求队列 """

    def __init__(self,
                 arrivals: List[float],
                 requests: List[Request],
                 workload_infos: Optional[Dict[str, Any]] = None):
        assert len(arrivals) == len(requests)

        self.arrivals = np.array(arrivals)
        self.requests = requests
        self.workload_infos = workload_infos
        
        if len(self.arrivals) > 1:
            intervals = self.arrivals[1:] - self.arrivals[:-1]
            self.rate = 1 / (np.mean(intervals) + eps)
            self.cv = np.std(intervals) * self.rate
        else:
            self.rate = 0
            self.cv = 0

    @staticmethod
    def from_workload_file(workload_file: str):
        """从make gen-workload生成的文件导入Workload"""
        with open(workload_file) as f:
            workload_json = json.load(f)
        arrivals = workload_json["arrivals"]
        requests = [Request(**r) for r in workload_json["requests"]]
        workload_infos = workload_json.get("info", None)
        # convert to numpy array
        for req in requests:
            if isinstance(req.data[0], list):
                req.data = (np.array(req.data[0]), req.data[1], req.data[2])
        return Workload(arrivals, requests, workload_infos)

    def split_round_robin(self, number: int):
        """将Workload轮询式拆分为number个小Workload"""
        rets = []
        for i in range(number):
            rets.append(self[i::number])
        return rets

    def split_time_interval(self, interval: float):
        """根据时间间隔划分Workload，间隔过大则拆开"""
        if len(self.arrivals) < 1:
            return []
        ws = []
        start_i = 0
        start_time = self.arrivals[start_i]
        for i in range(len(self.arrivals)):
            if self.arrivals[i] > start_time + interval:
                ws.append(self[start_i:i])
                start_i = i
                start_time = self.arrivals[i]
        ws.append(self[start_i:])
        return ws

    def num_model_requests(self, model_name: str):
        """统计指定模型的请求数量"""
        return len([r for r in self.requests if r.model_name == model_name])

    def split_by_model(self, model_name: str):
        """获取指定模型的Workload"""
        if len(self.arrivals) < 1:
            return []
        arrivals = []
        requests = []
        workload_infos = self.workload_infos
        for i in range(len(self.arrivals)):
            if self.requests[i].model_name == model_name:
                req = deepcopy(self.requests[i])
                req.idx = len(arrivals)
                arrivals.append(self.arrivals[i])
                requests.append(req)
        return Workload(arrivals, requests, workload_infos)

    def split_by_models(self, models: List[str]):
        """获取指定多个模型的混合Workload"""
        if len(self.arrivals) < 1:
            return []
        arrivals = []
        requests = []
        workload_infos = self.workload_infos
        for i in range(len(self.arrivals)):
            if self.requests[i].model_name in models:
                req = deepcopy(self.requests[i])
                req.idx = len(arrivals)
                arrivals.append(self.arrivals[i])
                requests.append(req)
        return Workload(arrivals, requests, workload_infos)

    def compute_stats(self, start: Sequence[float], finish: Sequence[float], good: Sequence[bool], warmup: float):
        """获取模型统计数据"""
        # 根据预热时间占比，裁剪掉部分首尾请求
        if len(self.arrivals) > 1:
            skip = int(warmup / (self.arrivals[-1] - self.arrivals[0]) * len(self.arrivals))
            if skip > 0:
                start = start[skip:-skip]
                finish = finish[skip:-skip]
                good = good[skip:-skip]
                requests = self.requests[skip:-skip]
        # 逐个模型统计
        model_indices = defaultdict(list) # 指定模型所有请求索引
        for i in range(len(requests)):
            model_indices[requests[i].model_name].append(i)
        names = list(model_indices.keys())
        names.sort(key=lambda name: len(model_indices[name])) # 按请求总数升序排列模型
        stats = []
        for name in names:
            indices = np.asarray(model_indices[name], dtype=np.int32)
            tmp_good = np.asarray(good[indices], dtype=bool)
            # 筛选满足SLO的请求
            tmp_start = start[indices][tmp_good]
            tmp_finish = finish[indices][tmp_good]
            # 计算吞吐量和延迟（P90/P95/P99）
            goodput = np.mean(tmp_good)
            if goodput > 0:
                throughput = len(tmp_start) / (tmp_start[-1] - tmp_start[0])
                latency = tmp_finish - tmp_start
            else:
                throughput = 0
                latency = [0]
            sorted_latency = np.sort(latency)
            latency_p90 = sorted_latency[int(0.90 * len(sorted_latency))]
            latency_p95 = sorted_latency[int(0.95 * len(sorted_latency))]
            latency_p99 = sorted_latency[int(0.99 * len(sorted_latency))]
            stats.append(
                PerModelStatsResult(
                    name, len(indices), goodput, throughput,
                    np.mean(latency), np.std(latency),
                    latency_p90, latency_p95, latency_p99, latency,
                    tmp_start, tmp_finish
                )
            )
        return StatsResult(stats, None, np.mean(good), np.mean(finish-start), len(start), len(start)/(start[-1]-start[0]))

    @staticmethod
    def print_stats(stats: StatsResult):
        """打印模型统计数据"""
        if stats.per_model_stats:
            print("--- per model ---")
            for stat in stats.per_model_stats:
                print(f"model: {stat.name}, #req: {stat.num_requests}")
                print(f"goodput: {stat.goodput*100:.2f} %, "
                      f"throughput: {stat.throughput:.2f} q/s, ")
                print(f"latency mean: {stat.latency_mean*1e3:.2f} ms, "
                      f"std: {stat.latency_std*1e3:.2f} ms, "
                      f"p90: {stat.latency_p90*1e3:.2f} ms, "
                      f"p95: {stat.latency_p95*1e3:.2f} ms, "
                      f"p99: {stat.latency_p99*1e3:.2f} ms, ")
        if stats.group_num_requests is not None:
            print(f"per group #req: {stats.group_num_requests}")
        print("--- overall ---")
        print(f"total #req: {stats.num_requests}, "
              f"rate: {stats.request_rate:.2f} q/s")
        print(f"average goodput: {stats.goodput*100:.2f} %, "
              f"latency mean: {stats.latency_mean*1e3:.2f} ms")

    @classmethod
    def empty(cls):
        """创建空对象"""
        return cls([], [])

    @classmethod
    def merge(cls, *args):
        """合成多个Workload，按请求到达时间重组"""
        if len(args) == 1:
            return args[0]
        number = sum(len(x) for x in args)
        merged_arrivals = np.concatenate(tuple(x.arrivals for x in args))
        merged_requests = sum((x.requests for x in args), []) # 列表拼接
        sorted_indices = np.argsort(merged_arrivals)
        arrivals = [None] * number
        requests = [None] * number
        for i, j in enumerate(sorted_indices):
            arrivals[i] = merged_arrivals[j]
            requests[i] = merged_requests[j]
            requests[i].idx = i
        return cls(arrivals, requests)

    def __getitem__(self, key):
        if isinstance(key, slice):
            arrivals = self.arrivals.__getitem__(key)
            requests = self.requests.__getitem__(key)
            return Workload(arrivals, requests)
        else:
            raise NotImplementedError()

    def __add__(self, other):
        return Workload.merge(self, other)

    def __len__(self):
        return len(self.arrivals)

    def __str__(self):
        return (f"Workload(len={len(self)}, "
                f"rate={self.rate:.2f}, "
                f"CV={self.cv:.2f}, "
                f"tstamps={to_str_round(self.arrivals[:20])} ...)")


def sample_requests(
    num_requests: int,
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase = None,
    seqlen_distribution: Optional[Tuple[int, int]] = None,
    tokenized_cache_path: str = DEFAULT_TOKENIZED_CACHE,
) -> List[Tuple[list[int], int, int]]:
    '''
    处理数据集并随机采样，获取请求序列
    return: [(encoded_prompt, intput_len, output_len) * num_requests]
    '''
    if tokenizer is None:
        tokenizer = get_tokenizer(DEFAULT_TOKENIZER_PATH)
        
    if tokenized_cache_path and os.path.exists(tokenized_cache_path):
        # 使用tokenize cache
        with open(tokenized_cache_path, "rb") as fp:
            tokenized_dataset: list[tuple[str, list[int], int]] = pickle.load(fp)
    else:
        # 导入数据集，筛选出所有前两句对话，第一句为prompt，第二句为completion
        with open(dataset_path) as f:
            dataset = json.load(f)
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        dataset = [(data["conversations"][0]["value"], data["conversations"][1]["value"]) for data in dataset]
        # Tokenize the prompts and completions
        prompts = [prompt for prompt, _ in dataset]
        prompt_token_ids = tokenizer(prompts).input_ids # prompt tokens
        completions = [completion for _, completion in dataset]
        completion_token_ids = tokenizer(completions).input_ids # completion tokens
        tokenized_dataset = []
        for i in range(len(dataset)):
            output_len = len(completion_token_ids[i])
            tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))
        if tokenized_cache_path: # 存入cache
            os.makedirs(os.path.dirname(tokenized_cache_path), exist_ok=True)
            with open(tokenized_cache_path, "wb") as fp:
                pickle.dump(tokenized_dataset, fp)

    min_seq_len = 0
    max_seq_len = np.inf
    if seqlen_distribution is not None:
        min_seq_len, max_seq_len = seqlen_distribution
    # 根据长度过滤对话.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for _, prompt_token_id, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_id)
        total_len = prompt_len + output_len
        if prompt_len < 4 or output_len < 4 or total_len < min_seq_len:
            # 过滤掉过短对话，避免TGI（Text Generation Inference）报错
            continue
        if prompt_len > 1024 or total_len > min(max_seq_len, 2048):
            # 过滤过长对话
            continue
        filtered_dataset.append((prompt_token_id, prompt_len, output_len))
    # 在处理后的数据集中采样
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


def generate_workload_requests(
    num_requests: int,
    prompt_len: int,
    output_len: int,
    tokenizer: PreTrainedTokenizerBase,
    distribution: str = "fixed"
):
    """ 随机生成请求（str），请求为固定或随机长度 """
    if distribution == "fixed":
        prompt_lens = [prompt_len] * num_requests
        output_lens = [output_len] * num_requests
    elif distribution == "uniform":
        # 0.5~1.5倍prompt_len均匀分布
        prompt_lens = np.random.uniform(prompt_len // 2, prompt_len + prompt_len // 2, size=num_requests)
        output_lens = np.random.uniform(output_len // 2, output_len + output_len // 2, size=num_requests)
    request_datasets = []
    for i in range(num_requests):
        cur_prompt_len = int(prompt_lens[i])
        cur_output_len = int(output_lens[i])
        prompt = np.random.randint(0, tokenizer.vocab_size, size=cur_prompt_len).tolist()
        request_datasets.append((prompt, cur_prompt_len, cur_output_len))
    return request_datasets


def get_workload(
    models: List[str],
    arrival_rates: List[float],
    start: int,
    duration: int,
    distribution: str = "poisson",
    seed: int = 0,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    num_requests: Optional[List[int]] = None,
    sampled_requests = None,
    prompt_distribution = None,
    prompt_lens = 0,
    output_lens = 0,
    trace = None,
) -> Workload:
    """给定模型和请求速率，生成Workload"""
    workloads = []
    for i, (model_name, arrival_rate, num_request) in enumerate(zip(models, arrival_rates, num_requests)):
        if trace is None:
            assert arrival_rate >= 0
            if distribution == "poisson":
                process = PoissonProcess
            elif distribution == "uniform":
                process = DeterministicProcess
            else:
                raise ValueError(f"Unknown arrival process: {distribution}")
            workload = process(arrival_rate).generate_workload(
                # 生成指定分布的Workload，Request.data均初始化为空
                model_name, start=start, duration=duration, num_requests=num_request, seed=seed
            )
            # 填充Request.data
            if sampled_requests is not None:
                for req_id in range(len(workload)):
                    workload.requests[req_id].data = sampled_requests[i][req_id]
            else:
                # 给定对话长度时，随机生成数据
                if isinstance(prompt_lens, int):
                    prompt_len = prompt_lens
                    output_len = output_lens
                else:
                    prompt_len = prompt_lens[i]
                    output_len = output_lens[i]
                workload = replace_long_workloads(
                    workload,
                    tokenizer_path,
                    prompt_len,
                    output_len,
                    distribution=prompt_distribution
                )
        else:
            workload = TraceProcess(trace).generate_workload(model_name, tokenizer_path)
        workloads.append(workload)
        seed += random.randint(1, 100)
    workload = Workload.merge(*workloads)
    return workload


def replace_long_workloads(
    workload: Workload,
    tokenizer_path,
    prompt_len,
    output_len,
    distribution: str = "fixed"
):
    """指定对话长度，随机生成数据"""
    tokenizer = get_tokenizer(tokenizer_path)
    num_requests = len(workload)
    workload_reqs = generate_workload_requests(num_requests, prompt_len, output_len, tokenizer, distribution=distribution)
    for i in range(num_requests):
        workload.requests[i].data = workload_reqs[i]
    return workload


def generate_workload(
    workload_infos,
    output_file,
    num_requests = 1500,
    start = 0,
    duration = 2000,
    distribution = "poisson",
    sampled_requests = None,
    prompt_distribution = None,
    prompt_len = 0,
    output_len = 0,
    trace = None,
):
    """获取Workload并写入json文件"""
    if isinstance(num_requests, int):
        num_requests = [num_requests] * len(workload_infos)
    models = [model for model, _ in workload_infos]
    arrival_rates = [rate for _, rate in workload_infos]
    workload = get_workload(
        models,
        arrival_rates,
        start,
        duration,
        num_requests=num_requests,
        sampled_requests=sampled_requests,
        prompt_distribution=prompt_distribution,
        prompt_lens=prompt_len,
        output_lens=output_len,
        trace = trace
    )
    workload_num_requests = []
    for model, _ in workload_infos:
        workload_num_requests.append(workload.num_model_requests(model))
    workload_json = {
        "info": {
            "rates": workload_infos,
            "start": start,
            "duration": duration,
            "num_requests": workload_num_requests,
            "distribution": distribution,
            "prompt_len": prompt_len,
            "output_len": output_len,
        },
        "arrivals": workload.arrivals.tolist(),
        "requests": [dataclasses.asdict(r) for r in workload.requests]
    }
    print(f"Save workload to {output_file}")
    with open(output_file, "w") as f:
        json.dump(workload_json, f)


def get_workloads_info_from_yaml(models_yaml: str) -> List[Tuple[str, float]]:
    """ 从yaml中读取(模型名称, 请求速率) """
    with open(models_yaml, "r") as fp:
        model_group = yaml.safe_load(fp)
    models = model_group["models"]
    model_id = [model["name"] for model in models]
    rate_list = [model["rate"] for model in models]
    return [(id, rate) for id, rate in zip(model_id, rate_list)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-source", type=str, default=None, help="the dataset source, like sharedgpt")
    parser.add_argument("--trace", type=str, default=None)
    parser.add_argument("--model-yaml", type=str, default="examples/basic/models.yaml", help="the model yaml to generate the workload, refer to `examples/basic/models.yaml`")
    parser.add_argument("--output-file", type=str, default=None, help="the dataset source, like sharedgpt")
    args = parser.parse_args()
    dataset = args.dataset_source
    trace = args.trace
    models_yaml = args.model_yaml
    output_file = args.output_file
    num_requests = 200
    start = 0
    duration = 1000
    distribution = "poisson"
    workload_infos = get_workloads_info_from_yaml(models_yaml)
    print(f"Get workload info from {models_yaml}:\n{workload_infos}")
    rate_dist = [v[1] for v in workload_infos]
    max_rate = max(rate_dist)
    capped_num_requests = min(num_requests, int(max_rate * duration)) # 单模型最大请求数量
    num_requests_dist = [capped_num_requests]
    dispatch_duration = capped_num_requests / max_rate * 1.02 # 最大负载模型的运行时间
    for rate in rate_dist[1:]: num_requests_dist.append(max(int(rate * dispatch_duration), 1)) # 为其他模型确定请求数量
    workload_infos = [(f"llm-{model_id}", rate) for model_id, rate in enumerate(rate_dist)]
    
    if dataset is not None and os.path.exists(dataset):
        sampled_requests = []
        for i in range(len(rate_dist)):
            # 按模型速率比例采样
            cur_num_requests = int(rate_dist[i] * num_requests * 1.1 / max_rate)
            sampled_requests.append(sample_requests(cur_num_requests, dataset))
        print(f"Sample total {sum(num_requests_dist)} requests")
        prompt_distribution=None
        prompt_len=None
        output_len=None
    else:
        sampled_requests = None
        prompt_distribution="uniform"
        prompt_len=[64, 16, 32]
        output_len=[16, 64, 32]
    
    generate_workload(
        workload_infos,
        output_file,
        num_requests=num_requests_dist,
        start=start,
        duration=duration,
        distribution=distribution,
        sampled_requests=sampled_requests,
        prompt_distribution=prompt_distribution,
        prompt_len=prompt_len,
        output_len=output_len,
        trace = trace
    )