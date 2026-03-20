import os
import argparse
import bisect
import json
import math
import yaml
import itertools
import cvxpy as cp
import numpy as np
from typing import Dict, List, Optional
from transformers import AutoConfig

MEMORY_PER_GPU = 80  # GB


class YamlBuilder:

    def __init__(self, ngpus, overload_threshold):
        self.data = {
            "num_gpus":
            ngpus,
            "max_num_seqs":
            256,  # default
            "overload_threshold":
            overload_threshold // 10 if overload_threshold else 2,
            "gpu_memory_utilization":
            0.5,  # TODO
            "models": [],
            "workloads": {
                "workload_file": None
            },
        }
        self.model_sizes = []
        yaml.Dumper.add_representer(
            type(None), lambda dumper, value: dumper.represent_scalar(
                u'tag:yaml.org,2002:null', ''))

    def add_model(self,
                  name,
                  model,
                  tensor_parallel_size,
                  placement,
                  mps_percentage,
                  max_num_seqs,
                  model_dtype="fp16"):
        model_data = {
            "name": name,
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": 1,
            "placement": placement,
            "mps_percentage": mps_percentage,
            "max_num_seqs": max_num_seqs,
            "model_dtype": model_dtype
        }
        self.data["models"].append(model_data)
        self.update_utilization(
            model, model_data["tensor_parallel_size"] *
            model_data["pipeline_parallel_size"])

    def update_utilization(self, model, ngpus):
        '''
        80: A100 HBM
        10: activation + reserved mem
        0.8: proc init mem, one model - two procs
        '''
        model_size = int(model.split("-")[-1][:-1]) * 2
        self.model_sizes.append(model_size)
        self.data["gpu_memory_utilization"] = (
            80 - sum(self.model_sizes) / ngpus -
            0.8 * len(self.model_sizes) * 2 - 6 * len(self.model_sizes)) / 80

    def build(self):
        return self.data

    def to_yaml(self):
        return yaml.dump(self.data, default_flow_style=False, sort_keys=False)

    def dump_to_file(self, file_path):

        if not self.data["models"]:
            print("no models, no dump")
            return

        self.data["workloads"]["workload_file"] = None

        with open(file_path, 'w') as file:
            yaml.dump(self.data, file, sort_keys=False)

        print(f"yaml save to {file_path}")


class CostEstimator:

    def __init__(self, model_set: List[str], gpu_memory_utilization: float):
        self.model_cost: Dict[str, Dict[int, Dict[int, Dict[int, float]]]] = {}
        self.gpu_memory_utilization = gpu_memory_utilization

    def add_model_cost(self, model: str,
                       model_cost: Dict[int, Dict[int, Dict[int, float]]]):
        self.model_cost[model] = {}
        for ngpu, ngpu_cost in model_cost.items():
            ngpu = int(ngpu)
            self.model_cost[model][ngpu] = {}
            for mps, mps_cost in ngpu_cost.items():
                mps = int(mps)
                self.model_cost[model][ngpu][mps] = {}
                for batch_size, bs_cost in mps_cost.items():
                    batch_size = int(batch_size)
                    self.model_cost[model][ngpu][mps][batch_size] = {}
                    decoding_latencies = []
                    for seq_len, cost_info in bs_cost.items():
                        seq_len = int(seq_len)
                        prefill_cost = cost_info["prefill"]
                        decode_cost = cost_info["decoding"]
                        if prefill_cost < 0 and decode_cost < 0:
                            continue

                        self.model_cost[model][ngpu][mps][batch_size][
                            seq_len] = {}
                        if prefill_cost > 0:
                            self.model_cost[model][ngpu][mps][batch_size][
                                seq_len]["prefill"] = prefill_cost / 1000
                        if decode_cost > 0:
                            decoding_latencies.append(decode_cost)
                            self.model_cost[model][ngpu][mps][batch_size][
                                seq_len]["decoding"] = np.mean(
                                    decoding_latencies) / 1000

    def get_batch_cost_info(self,
                            model: str,
                            ngpu: int,
                            mps: int,
                            is_prefill: bool = False,
                            avg_seq_len: int = 512):
        costs = self.model_cost[model][ngpu][mps]
        key = "prefill" if is_prefill else "decoding"
        bs_list, lats = [], []
        for bs in costs:
            seq_lats = []
            for seq_len, _ in costs[bs].items():
                if key not in costs[bs][seq_len]:
                    continue
                seq_lats.append(costs[bs][seq_len][key])
                if seq_len > avg_seq_len:
                    break
            if len(seq_lats) > 0:
                bs_list.append(bs)
                lats.append(seq_lats[-1] if is_prefill else np.mean(seq_lats))
        return bs_list, lats

    def get_latency(self, model: str, bs: int, prefill_bs: List[int],
                    prefill_lats: List[float], decoding_bs: List[int],
                    decoding_lats: List[float]):
        i = bisect.bisect_left(decoding_bs, bs)
        # interpolation decoding latency
        t_decode = np.interp(bs, decoding_bs[i:i + 2], decoding_lats[i:i + 2])
        t_prefill = np.interp(bs, prefill_bs[i:i + 2], prefill_lats[i:i + 2])

        # add schedule overhead
        if "llama-7b" in model:
            t_decode += 0.0001 * bs
        elif "llama-13b" in model:
            t_decode += 0.00008 * bs
        else:
            t_decode += 0.00007 * bs
        return t_prefill, t_decode

    def estimate_throughput(
        self,
        model: str,
        ngpu: int,
        rate: int,
        decoding_mps: int,
        avg_output_len: int,
        kv_cache_size_per_token: float,
        prefill_mps: int = 100,
        avg_prompt_len: int = 128,
        cache_size: Optional[float] = None,
        return_latency: bool = False,
    ):
        prefill_bs, prefill_lats = self.get_batch_cost_info(
            model,
            ngpu,
            prefill_mps,
            is_prefill=True,
            avg_seq_len=avg_prompt_len)
        decoding_bs, decoding_lats = self.get_batch_cost_info(
            model,
            ngpu,
            decoding_mps,
            is_prefill=False,
            avg_seq_len=avg_output_len)

        # How to calaulate the throughput: P
        #    estimate average sequence output length: S
        #    estimate average batch size: B
        #    estimate output latency per output token: T_decoding
        #    estimate prefll: T_prefill
        # We have the following equation:
        #    S * T_decoding * P = B
        # We can estimate the throughput P = B / (T_prefill + S * T_decoding)
        kv_cache_per_seq = kv_cache_size_per_token * (avg_output_len +
                                                      avg_prompt_len)
        if cache_size is None:
            cache_size = self.gpu_memory_utilization * MEMORY_PER_GPU * ngpu
        max_num_seqs = min(np.floor_divide(cache_size, kv_cache_per_seq),
                           max(decoding_bs))

        # find the suitable batch size that can satisfy the throughput
        # with binary search
        tpt = 0
        low_bs, high_bs = 1, max(2, max_num_seqs)
        while True:
            if high_bs - low_bs < 1:
                break
            cur_bs = (low_bs + high_bs) / 2
            t_prefill, t_decode = self.get_latency(model, cur_bs, prefill_bs,
                                                   prefill_lats, decoding_bs,
                                                   decoding_lats)

            tpt = cur_bs / (t_decode * avg_output_len + t_prefill)
            if tpt >= rate:
                high_bs = cur_bs
            else:
                low_bs = cur_bs

        can_satisfy = False
        if tpt + min(0.5, tpt * 0.1) > rate:
            can_satisfy = True
            tpt = rate
        if return_latency:
            return can_satisfy, tpt, int(cur_bs), t_decode, t_prefill
        return can_satisfy, tpt, int(cur_bs)

    def estimate_mps(
        self,
        model: str,
        ngpu: int,
        rate: int,
        avg_output_len: int,
        kv_cache_size_per_token: float,
        prefill_mps: int = 100,
        avg_prompt_len: int = 128,
    ):
        # How to calaulate the throughput
        #    estimate average sequence output length: S
        #    estimate running time of mps: T
        #    estimate request throughput: BS / (S * T)
        #
        model = model.split("/")[-1]
        # unsupported (model, ngpu) pair
        if ngpu not in self.model_cost[model]:
            return False, 0, 0, 0
        max_tpt_config = (False, None, None, 0)
        for mps in sorted(self.model_cost[model][ngpu]):
            if mps > 90 or mps == prefill_mps:
                continue
            can_satisfy, expected_tpt, expected_bs = self.estimate_throughput(
                model, ngpu, rate, mps, avg_output_len,
                kv_cache_size_per_token, prefill_mps, avg_prompt_len)
            if can_satisfy:
                return True, mps, expected_bs, expected_tpt
            if expected_tpt > max_tpt_config[3]:
                max_tpt_config = (False, mps, expected_bs, expected_tpt)
        return max_tpt_config

    def estimate_kv_cache_size(self, mesh, llm_list):
        weight = sum([llm.model_size for llm in llm_list])
        activation = len(llm_list) * 6
        kv_cache_size = mesh.gpu_memory * 0.98 - weight - activation
        return max(0, kv_cache_size)

    def estimate_mesh_throughput(self,
                                 mesh,
                                 llm_list,
                                 decoding_mps_list,
                                 prefill_mps_list=None):
        # split kv cache
        total_base = sum([llm.token_kv_size * llm.rate for llm in llm_list])
        kv_cache_size = self.estimate_kv_cache_size(mesh, llm_list)

        batch_sizes, max_num_seqs = [], []
        llm_costs = {}
        for i, llm in enumerate(llm_list):
            local_cache_size = llm.rate * llm.token_kv_size / total_base * kv_cache_size
            avg_output_len = llm.avg_output_len
            avg_prompt_len = llm.avg_prompt_len
            prefill_mps = prefill_mps_list[i] if prefill_mps_list else 100
            decoding_mps = decoding_mps_list[i]

            model = llm.model.split("/")[-1]
            ret = self.estimate_throughput(model, mesh.ngpus, llm.rate,
                                           decoding_mps, avg_output_len,
                                           llm.token_kv_size, prefill_mps,
                                           avg_prompt_len, local_cache_size)
            can_satisfy, _, batch_size = ret
            batch_sizes.append(batch_size)

            prefill_bs, prefill_lats = self.get_batch_cost_info(
                model,
                mesh.ngpus,
                prefill_mps,
                is_prefill=True,
                avg_seq_len=avg_prompt_len)
            decoding_bs, decoding_lats = self.get_batch_cost_info(
                model,
                mesh.ngpus,
                decoding_mps,
                is_prefill=False,
                avg_seq_len=avg_output_len)
            llm_costs[llm.name] = {
                "prefill": (prefill_bs, prefill_lats),
                "decoding": (decoding_bs, decoding_lats)
            }
            kv_cache_per_seq = llm.token_kv_size * (avg_output_len +
                                                    avg_prompt_len)
            max_bs = min(np.floor_divide(local_cache_size, kv_cache_per_seq),
                         max(decoding_bs))
            max_num_seqs.append(max_bs)

        # adjusted batch size due to multiplexing
        est_tpts = []
        max_tpt_bs, max_tpt = None, 0
        while True:
            end = True
            for i, bs in enumerate(batch_sizes):
                if bs < max_num_seqs[i]:
                    end = False
                    break

            if end:
                break

            t_decodes, t_prefills = [], []
            for llm, cur_bs in zip(llm_list, batch_sizes):
                prefill_bs, prefill_lats = llm_costs[llm.name]["prefill"]
                decoding_bs, decoding_lats = llm_costs[llm.name]["decoding"]

                model = llm.model.split("/")[-1]
                t_prefill, t_decode = self.get_latency(model, cur_bs,
                                                       prefill_bs,
                                                       prefill_lats,
                                                       decoding_bs,
                                                       decoding_lats)
                t_decodes.append(t_decode)
                t_prefills.append(t_prefill)

            mesh_tpt = 0
            min_achieved_llm, min_achieved_ratio = None, 100
            est_tpts = []
            for i, llm in enumerate(llm_list):
                tpt = batch_sizes[i] / (t_decodes[i] * llm.avg_output_len +
                                        sum(t_prefills))
                est_tpts.append(tpt)

                mesh_tpt += tpt * llm.rate
                if tpt + min(0.5, tpt * 0.1) < llm.rate:
                    achieved_ratio = tpt / llm.rate
                    if batch_sizes[i] >= max_num_seqs[i]:
                        continue
                    if achieved_ratio < min_achieved_ratio:
                        min_achieved_ratio = achieved_ratio
                        min_achieved_llm = i
            mesh_tpt = mesh_tpt / sum([llm.rate for llm in llm_list])
            if max_tpt < mesh_tpt:
                max_tpt = mesh_tpt
                max_tpt_bs = batch_sizes.copy()

            if min_achieved_llm is not None:
                if batch_sizes[min_achieved_llm] < max_num_seqs[
                        min_achieved_llm]:
                    batch_sizes[min_achieved_llm] += 1
            else:
                break

        # print(f"Mesh size: {mesh.ngpus}, Max tpt: {max_tpt:.3f}")
        # for i, llm in enumerate(llm_list):
        #     print(f"  LLM: {llm.name}, Model: {llm.model.split('/')[-1]}, "
        #           f"Rate: {llm.rate}, Batch size: {max_tpt_bs[i]}, "
        #           f"Throughput: {est_tpts[i]:.3f}, "
        #           f"Max num seq: {max_num_seqs[i]}")
        return max_tpt, max_tpt_bs, est_tpts


class LLM:

    def __init__(
        self,
        name: str,
        model: str,
        rate: int,
        avg_output_len: int,
        prefill_mps: int = 100,
        avg_prompt_len: int = 128,
    ):
        self.name = name
        self.model = model
        self.rate = rate
        self.avg_output_len = avg_output_len
        self.prefill_mps = prefill_mps
        self.avg_prompt_len = avg_prompt_len

        self.model_size = int(self.model.split("-")[-1][:-1]) * 2

        self.base_ngpu = 1
        if "llama-30b" in self.model:
            self.base_ngpu = 2
        elif "llama-65b" in self.model:
            self.base_ngpu = 4

        self.rate_scale = 1
        if "llama-13b" in self.model:
            self.rate_scale = 2
        elif "llama-30b" in self.model:
            self.rate_scale = 4
        elif "llama-65b" in self.model:
            self.rate_scale = 8

        self.config = AutoConfig.from_pretrained(self.model)
        self.num_hidden_layers = self.config.num_hidden_layers
        self.num_attention_heads = self.config.num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.token_kv_size = self.num_hidden_layers * self.hidden_size * 2
        self.token_kv_size = self.token_kv_size * 2 / (1024**3)

    def set_candidate_mesh_size(
        self,
        cost_estimator: CostEstimator,
        avg_output_len: int,
        prefill_mps: int = 100,
        avg_prompt_len: int = 128,
        verbose: bool = False,
    ):
        self.mesh_candidates = []
        self.min_mesh_size = 8
        for ngpu in [1, 2, 4, 8]:
            if ngpu < self.base_ngpu:
                continue
            ret = cost_estimator.estimate_mps(self.model, ngpu, self.rate,
                                              avg_output_len,
                                              self.token_kv_size, prefill_mps,
                                              avg_prompt_len)
            (can_satisfy, mps, batch_size, expected_tpt) = ret
            if mps <= 0:
                continue
            # if not can_satisfy or mps > 60:
            #     continue
            self.mesh_candidates.append(
                (ngpu, mps, batch_size, expected_tpt, can_satisfy))
            self.min_mesh_size = min(self.min_mesh_size, ngpu)
        assert len(self.mesh_candidates) > 0, "No valid mesh size"

        if verbose:
            print(f"## LLM: {self.name}, Model: {self.model}, "
                  f"Rate: {self.rate}, Candidates:")
            for i in range(len(self.mesh_candidates)):
                candidate = self.mesh_candidates[i]
                ngpu, mps, batch_size, expected_tpt, can_satisfy = candidate
                print(f"##   ngpu: {ngpu}, mps: {mps}, bs: {batch_size}, tpt: "
                      f"{expected_tpt:.3f}, can_satisfy: {can_satisfy}")

    def query_candidate_tpt(self, ngpu: int):
        for candidate in self.mesh_candidates:
            if candidate[0] == ngpu:
                return candidate[2], candidate[3]
        return 0, 0

    def query_candidate_mps(self, ngpu: int):
        for candidate in self.mesh_candidates:
            if candidate[0] == ngpu:
                return candidate[1]
        return 0

    def query_candidate(self, ngpu: int):
        for candidate in self.mesh_candidates:
            if candidate[0] == ngpu:
                return candidate
        return None

    def __repr__(self):
        return (f"LLM("
                f"name={repr(self.name)}, "
                f"model={repr(self.model)}, "
                f"rate={repr(self.rate)}, "
                f"base_ngpu={repr(self.base_ngpu)}, "
                f"rate_scale={repr(self.rate_scale)}, "
                f"num_hidden_layers={repr(self.num_hidden_layers)}, "
                f"num_attention_heads={repr(self.num_attention_heads)}, "
                f"hidden_size={repr(self.hidden_size)}, "
                f"token_kv_size={repr(self.token_kv_size)},"
                f"min_mesh_size={repr(self.min_mesh_size)},"
                f"mesh_candidates={repr(self.mesh_candidates)},"
                f")")


class MeshGroup:

    def __init__(self, ngpus: int, gpu_memory_utilization: float = 0.4):
        self.ngpus = ngpus
        self.overload_threshold = 20
        self.remaining_mps = 100
        self.overloaded = False
        self.models: list[tuple[LLM, int]] = []

        self.gpu_memory = MEMORY_PER_GPU * self.ngpus
        self.gpu_memory_utilization = gpu_memory_utilization
        self.free_gpu_memory = 0.9 * (
            1 - self.gpu_memory_utilization) * self.gpu_memory

    @property
    def is_idle(self):
        return len(self.models) == 0

    def can_place(self, model: LLM, mps: int):
        if self.free_gpu_memory < model.model_size:
            return False
        if self.remaining_mps < mps:
            if self.remaining_mps + self.overload_threshold < mps:
                return False
        return True

    def place_model(self, model: LLM, mps: int):
        # first check memory
        if self.free_gpu_memory < model.model_size:
            return False
        if self.remaining_mps < mps:
            if self.remaining_mps + self.overload_threshold < mps:
                return False
            self.overloaded = True
        self.models.append((model, mps))
        self.free_gpu_memory -= model.model_size + 1.6 + 6
        self.remaining_mps -= mps
        return True

    def can_split(self):
        if len(self.models) > 0:
            return False
        # if self.ngpus % 2 != 0:
        # return False
        return True

    def split(self, ngpu=None):
        assert len(self.models) == 0, "MeshGroup is not empty"
        assert self.ngpus > 1, "MeshGroup size must be greater than 1"
        if ngpu is not None:
            ngpus_after_split = ngpu, self.ngpus - ngpu
        else:
            ngpus_after_split = self.ngpus // 2, self.ngpus // 2
        return MeshGroup(ngpus_after_split[0]), MeshGroup(ngpus_after_split[1])

    def __repr__(self) -> str:
        return (f"MeshGroup("
                f"ngpus={repr(self.ngpus)}, "
                f"gpu_memory_utilization={repr(self.gpu_memory_utilization)}, "
                f"overload_threshold={repr(self.overload_threshold)}, "
                f"remaining_mps={repr(self.remaining_mps)}, "
                f"overloaded={repr(self.overloaded)}, "
                f"models={repr(self.models)}, "
                f"gpu_memory={repr(self.gpu_memory)}, "
                f"free_gpu_memory={repr(self.free_gpu_memory)}"
                f")")


class Cluster:

    def __init__(self,
                 nnodes: int,
                 ngpus_per_node: int,
                 gpu_memory_utilization: float,
                 mesh_groups: Optional[List[MeshGroup]] = None):
        self.nnodes = nnodes
        self.ngpus_per_node = ngpus_per_node
        self.gpu_memory_utilization = gpu_memory_utilization

        if mesh_groups is None:
            mesh_groups = [
                MeshGroup(ngpus_per_node, gpu_memory_utilization)
                for _ in range(nnodes)
            ]
        self.mesh_groups = mesh_groups

    def generate_candidate_clusters(self):
        split_candidate = []
        if self.ngpus_per_node == 8:
            split_candidate.append([8])
            split_candidate.append([4, 4])
            split_candidate.append([4, 2, 2])
            split_candidate.append([4, 2, 1, 1])
            split_candidate.append([4, 1, 1, 1, 1])
            split_candidate.append([2, 2, 2, 2])
            split_candidate.append([2, 2, 2, 1, 1])
            split_candidate.append([2, 2, 1, 1, 1, 1])
            split_candidate.append([2, 1, 1, 1, 1, 1, 1])
            split_candidate.append([1, 1, 1, 1, 1, 1, 1, 1])
        elif self.ngpus_per_node == 6:
            split_candidate.append([4, 2])
            split_candidate.append([4, 1, 1])
            split_candidate.append([2, 2, 2])
            split_candidate.append([2, 2, 1, 1])
            split_candidate.append([2, 1, 1, 1, 1])
            split_candidate.append([1, 1, 1, 1, 1, 1])
        elif self.ngpus_per_node == 4:
            split_candidate.append([4])
            split_candidate.append([2, 2])
            split_candidate.append([2, 1, 1])
            split_candidate.append([1, 1, 1, 1])
        elif self.ngpus_per_node == 2:
            split_candidate.append([2])
            split_candidate.append([1, 1])
        elif self.ngpus_per_node == 1:
            split_candidate.append([1])

        for parallel_candidate in itertools.combinations_with_replacement(
                split_candidate, self.nnodes):
            candidate_clusters = []
            max_mesh_size = 1
            for node_mesh in parallel_candidate:
                max_mesh_size = max(max_mesh_size, max(node_mesh))
                for mesh_size in node_mesh:
                    candidate_clusters.append(MeshGroup(mesh_size))
            yield candidate_clusters, max_mesh_size


class PlacementOptimizer:

    def __init__(self,
                 workload_file: str,
                 cost_file: str,
                 rate_dict: Optional[Dict[str, float]] = None,
                 verbose: bool = False):
        self.workload_file = workload_file
        self.cost_file = cost_file

        with open(self.workload_file, "r") as f:
            model_group = yaml.safe_load(f)
        self.nnodes = model_group["cluster"]["nnodes"]
        self.ngpus_per_node = model_group["cluster"]["ngpus_per_node"]
        self.models: Dict[str, LLM] = {}

        avg_output_len = 337
        avg_prompt_len = 161

        total_memory_occupied = 0
        model_set = set()
        for model_cfg in model_group["models"]:
            if rate_dict is not None:
                rate = rate_dict[model_cfg["name"]]
            else:
                rate = model_cfg["rate"]
            llm = LLM(model_cfg["name"],
                      model_cfg["model"],
                      rate,
                      avg_output_len=avg_output_len,
                      prefill_mps=100,
                      avg_prompt_len=avg_prompt_len)
            self.models[llm.name] = llm
            total_memory_occupied += llm.model_size
            model_set.add(llm.model)
        # account activation memory
        total_memory_occupied += len(self.models) * 6
        avg_gpu_memory_utilization = total_memory_occupied / (
            MEMORY_PER_GPU * self.nnodes * self.ngpus_per_node)

        self.cluster = Cluster(self.nnodes, self.ngpus_per_node,
                               avg_gpu_memory_utilization)
        self.build_cost_model(model_set, avg_gpu_memory_utilization)

        for llm in self.models.values():
            llm.set_candidate_mesh_size(self.cost_estimator,
                                        avg_output_len,
                                        prefill_mps=100,
                                        avg_prompt_len=avg_prompt_len,
                                        verbose=verbose)

    def build_cost_model(self, model_set: List[str],
                         gpu_memory_utilization: float):
        with open(self.cost_file, "r") as f:
            cost_model = json.load(f)

        self.cost_estimator = CostEstimator(model_set, gpu_memory_utilization)
        for model, model_cost in cost_model.items():
            self.cost_estimator.add_model_cost(model, model_cost)

    def naive_greedy_placement(
        self,
        sorted_llm_list: List[LLM],
        mesh_list: List[MeshGroup],
        avg_output_len: int,
    ):
        # sort by `rate * compue_scale` ascending order => popular large model in the end
        sorted_llm_list = sorted(sorted_llm_list,
                                 key=lambda x:
                                 (x.rate * x.rate_scale, x.num_hidden_layers))
        mesh_list_with_tag = [(mesh, 0) for mesh in mesh_list]
        # sort by `used_computation_resource`(primary key), and `free_gpu_mem`
        f_mesh_sort = lambda x: (x[1], x[0].free_gpu_memory)
        mesh_list_with_tag.sort(key=f_mesh_sort, reverse=True)

        while len(sorted_llm_list) != 0:
            llm = sorted_llm_list[-1]
            for i, mesh_with_tag in enumerate(mesh_list_with_tag):
                mesh, comp_tag = mesh_with_tag
                mps = llm.query_candidate_mps(mesh.ngpus)
                if mps == 0:
                    return None
                if mesh.place_model(llm, mps):
                    amortized_comp = (llm.rate * llm.rate_scale) / mesh.ngpus
                    mesh_list_with_tag[i] = mesh, comp_tag - amortized_comp
                    break
            else:
                return None

            mesh_list_with_tag.sort(key=f_mesh_sort, reverse=True)
            sorted_llm_list.pop()

        next_mesh_list = [x[0] for x in mesh_list_with_tag]
        next_mesh_list = sorted(mesh_list, key=lambda x: x.ngpus, reverse=True)
        mesh_list_gpus = [mesh.ngpus for mesh in mesh_list]

        total_tpt = 0
        print(f"============= Greedy Placement =============")
        print(f"Find best mesh group: {mesh_list_gpus}")
        for mesh_idx, mesh in enumerate(next_mesh_list):
            print(f"  Mesh size: {mesh.ngpus}")
            for model_idx, (llm, mps) in enumerate(mesh.models):
                model_name = llm.model.split("/")[-1]
                batch_size, expected_tpt = llm.query_candidate_tpt(mesh.ngpus)
                total_tpt += expected_tpt
                print(f"    LLM: {llm.name}, Model: {model_name}, MPS: {mps}, "
                      f"rate: {llm.rate}, batch_size: {batch_size}, "
                      f"expected_tpt: {expected_tpt:.3f}")

        return mesh_list, total_tpt

    def smart_placement(self,
                        sorted_llm_list: List[LLM],
                        mesh_list: List[MeshGroup],
                        avg_output_len: int,
                        verbose: bool = False):
        if verbose:
            print(f"Search For Mesh List: "
                  f"{[mesh.ngpus for mesh in mesh_list]}")
        # sort by `rate * compue_scale` ascending order => popular large model in the end
        sorted_llm_list = sorted(sorted_llm_list,
                                 key=lambda x:
                                 (x.rate * x.rate_scale, x.rate_scale),
                                 reverse=True)
        mesh_list_with_tag = [(mesh, 0) for mesh in mesh_list]
        # sort by `used_computation_resource`(primary key), and `free_gpu_mem`
        f_mesh_sort = lambda x: (x[1], x[0].free_gpu_memory)
        mesh_list_with_tag.sort(key=f_mesh_sort, reverse=True)

        aggregate_tpt = 0
        mesh_throughputs = {i: [0, ()] for i in range(len(mesh_list))}
        for i, llm in enumerate(sorted_llm_list):
            best_mesh_id, best_tpt_increase = -1, -100
            for i, mesh_with_tag in enumerate(mesh_list_with_tag):
                mesh, comp_tag = mesh_with_tag
                decoding_mps = llm.query_candidate_mps(mesh.ngpus)
                if decoding_mps <= 0:
                    continue
                if not mesh.can_place(llm, decoding_mps):
                    continue

                llms_on_mesh = [m[0] for m in mesh.models]
                decoding_mps_list = [m[1] for m in mesh.models]
                llms_on_mesh.append(llm)
                decoding_mps_list.append(decoding_mps)

                tpt, _, _ = self.cost_estimator.estimate_mesh_throughput(
                    mesh, llms_on_mesh, decoding_mps_list)
                tpt_increase = tpt - mesh_throughputs[i][0]
                if tpt_increase > best_tpt_increase:
                    best_tpt_increase = tpt_increase
                    best_mesh_id = i
                elif tpt_increase == best_tpt_increase:
                    if mesh.ngpus < mesh_list[best_mesh_id].ngpus:
                        best_mesh_id = i
            if best_mesh_id >= 0:
                mesh, compute_tag = mesh_list_with_tag[best_mesh_id]
                amortized_comp = (llm.rate * llm.rate_scale) / mesh.ngpus
                mesh_list_with_tag[
                    best_mesh_id] = mesh, compute_tag - amortized_comp
                decoding_mps = llm.query_candidate_mps(mesh.ngpus)
                mesh.place_model(llm, decoding_mps)
                mesh_throughputs[best_mesh_id][0] += best_tpt_increase
            else:
                return None, -1

            mesh_list_with_tag.sort(key=f_mesh_sort, reverse=True)

        mesh_list = sorted(mesh_list, key=lambda x: x.ngpus, reverse=True)
        mesh_list_gpus = [mesh.ngpus for mesh in mesh_list]

        total_tpt, total_rate = 0, 0
        if verbose:
            print(f"============= Smart Placement =============")
            print(f"Find best mesh group: {mesh_list_gpus}")
        for mesh_idx, mesh in enumerate(mesh_list):
            llms_on_mesh = [m[0] for m in mesh.models]
            decoding_mps_list = [m[1] for m in mesh.models]
            mesh_tpt, batch_sizes, llm_tpts = self.cost_estimator.estimate_mesh_throughput(
                mesh, llms_on_mesh, decoding_mps_list)
            total_rate += sum([llm.rate for llm in llms_on_mesh])
            total_tpt += mesh_tpt * sum([llm.rate for llm in llms_on_mesh])
            if verbose:
                print(f"  Mesh size: {mesh.ngpus}, Mesh TPT: {mesh_tpt:.3f}")
            for model_idx, (llm, mps) in enumerate(mesh.models):
                model_name = llm.model.split("/")[-1]
                if verbose:
                    print(f"    LLM: {llm.name}, Model: {model_name}, "
                          f"MPS: {mps}, rate: {llm.rate}, batch_size: "
                          f"{batch_sizes[model_idx]}, "
                          f"expected_tpt: {llm_tpts[model_idx]:.3f}")
        total_tpt = total_tpt / total_rate
        return mesh_list, total_tpt

    def optimize(self,
                 is_greedy=False,
                 verbose: bool = True,
                 dump_to_yaml: bool = True,
                 dump_dir: str = None,
                 avg_output_len: int = 337,
                 avg_prompt_len: int = 161):
        # sharegpt data
        llm_list = sorted(self.models.values(),
                          key=lambda x: (x.rate * x.base_ngpu, x.base_ngpu),
                          reverse=True)
        max_mesh_size_required = max(llm.min_mesh_size for llm in llm_list)

        best_tpt, best_placement = -1, None
        candidate_gen = self.cluster.generate_candidate_clusters()
        for parallel_candidate, max_mesh_size in candidate_gen:
            if max_mesh_size < max_mesh_size_required:
                continue
            if len(parallel_candidate) > len(llm_list):
                continue

            if is_greedy:
                placement = self.naive_greedy_placement(
                    llm_list, parallel_candidate, avg_output_len)
            else:
                placement = self.smart_placement(llm_list,
                                                 parallel_candidate,
                                                 avg_output_len,
                                                 verbose=verbose)

            if placement is not None:
                mesh_list, opt_tpt = placement
                if opt_tpt > best_tpt:
                    best_tpt = opt_tpt
                    best_placement = mesh_list

        if best_placement is None:
            if verbose:
                print(f"Optimizer Done")
                print(f"Find no placement")
            return None

        mesh_list_gpus = [mesh.ngpus for mesh in best_placement]
        # {mesh_id: {llm_id: {model_type, expected_tpt}, best_tpt }
        ret = {"muxserve_tpt": best_tpt}
        sorted_placement = sorted(best_placement, key=lambda x: x.ngpus)
        idx = 0
        for i, mesh in enumerate(sorted_placement):
            if i == 0 or mesh.ngpus != sorted_placement[i - 1].ngpus:
                idx = 0
            else:
                idx += 1

            mesh_id = f"mesh{mesh.ngpus}idx{idx}"
            llms = {}
            for llm, _ in mesh.models:
                llms[llm.name] = {
                    "model_type": llm.model.split("/")[-1],
                    "expected_tpt": llm.query_candidate_tpt(mesh.ngpus)[-1],
                    "rate": llm.rate,
                }
            ret[mesh_id] = llms
        if dump_to_yaml:
            print(f"============= Optimizer Done =============")
            print(f"Find best mesh group: {mesh_list_gpus}")
            print(f"  Optimal throughput: {best_tpt}")
            self.dump_to_yaml(best_placement, mesh_list_gpus, is_greedy,
                              dump_dir)

        return ret

    @staticmethod
    def merge_idle_mesh(mesh_list: List[MeshGroup]):

        idle_gpu_num = sum([
            mesh_group.ngpus for mesh_group in mesh_list if mesh_group.is_idle
        ])
        not_idle = [instance for instance in mesh_list if not instance.is_idle]

        return not_idle + MeshGroup(idle_gpu_num)

    @staticmethod
    def dump_info(mesh_list: List[MeshGroup], info=''):
        mesh_list_gpus = [mesh.ngpus for mesh in mesh_list]
        print(f"============= {info} =============")
        print(f"Find best mesh group: {mesh_list_gpus}")
        for mesh_idx, mesh in enumerate(mesh_list):
            print(f"  Mesh size: {mesh.ngpus}")
            for model_idx, (llm, mps) in enumerate(mesh.models):
                model_name = llm.model.split("/")[-1]
                batch_size, expected_tpt = llm.query_candidate_tpt(mesh.ngpus)
                print(f"    LLM: {llm.name}, Model: {model_name}, MPS: {mps}, "
                      f"rate: {llm.rate}, batch_size: {batch_size}, "
                      f"expected_tpt: {expected_tpt:.3f}")

    def dump_to_yaml(self,
                     placement,
                     mesh_list_gpus,
                     is_greedy,
                     dump_dir=None):
        for mesh_idx, mesh in enumerate(placement):
            print(f"  Mesh size: {mesh.ngpus}")

            yaml_builder = YamlBuilder(
                ngpus=mesh.ngpus,
                overload_threshold=mesh.overload_threshold
                if mesh.overloaded else 0)

            for model_idx, (llm, mps) in enumerate(mesh.models):
                model_name = llm.model.split("/")[-1]
                batch_size, expected_tpt = llm.query_candidate_tpt(mesh.ngpus)
                print(f"    LLM: {llm.name}, Model: {model_name}, MPS: {mps}, "
                      f"rate: {llm.rate}, batch_size: {batch_size}, "
                      f"expected_tpt: {expected_tpt:.3f}")

                prefill_mps_dict = {
                    "llama-7b": 80,
                    "llama-13b": 90,
                    "llama-30b": 100,
                    "llama-65b": 100,
                }

                yaml_builder.add_model(
                    name=llm.name,
                    model=llm.model,
                    tensor_parallel_size=mesh.ngpus,
                    placement=[list(range(mesh.ngpus))],
                    mps_percentage=[prefill_mps_dict[model_name], mps],
                    max_num_seqs=batch_size,
                    model_dtype="fp16",
                )

            # print(yaml_builder.to_yaml())
            fname_prefix = os.path.splitext(
                os.path.basename(self.workload_file))[0]
            dump_dir = os.path.dirname(
                self.workload_file) if dump_dir is None else dump_dir
            fname = f"{fname_prefix}{'_Greedy' if is_greedy else ''}_GPUnum{sum(mesh_list_gpus)}_mesh_size{mesh.ngpus}_idx{mesh_idx}.yaml"
            yaml_builder.dump_to_file(os.path.join(dump_dir, fname), )

    def greedy_memory_placement(self,
                                dump_to_yaml: bool = True,
                                dump_dir=None):
        '''
        naive memory-greedy
        '''
        # sharegpt data
        avg_output_len = 337
        avg_prompt_len = 161

        for llm in self.models.values():
            llm.set_candidate_mesh_size(self.cost_estimator,
                                        avg_output_len,
                                        prefill_mps=100,
                                        avg_prompt_len=avg_prompt_len)
        llm_list = sorted(self.models.values(),
                          key=lambda x:
                          (x.min_mesh_size, x.rate * x.rate_scale),
                          reverse=True)

        mesh_list = [MeshGroup(self.ngpus_per_node)] * self.nnodes
        print(repr(mesh_list))
        for llm in llm_list:
            mesh_list = sorted(mesh_list,
                               key=lambda x: (1, -x.ngpus) if x.is_idle else
                               (0, x.free_gpu_memory / x.ngpus),
                               reverse=True)
            find_mesh_placement = False
            next_mesh_list = []

            for mesh in mesh_list:
                if mesh.ngpus < llm.min_mesh_size or find_mesh_placement:
                    next_mesh_list.append(mesh)
                    continue

                selected_mesh = mesh
                if mesh.ngpus > llm.min_mesh_size and mesh.can_split():
                    llm_need_gpus = llm.min_mesh_size
                    while llm_need_gpus < mesh.ngpus:
                        candidate = llm.query_candidate(llm_need_gpus)
                        ngpu, mps, batch_size, expected_tpt, can_satisfy = candidate
                        # if can_satisfy and mps < 60:
                        if can_satisfy:
                            sub_meshes = mesh.split(llm_need_gpus)
                            selected_mesh = sub_meshes[0]
                            next_mesh_list.append(sub_meshes[1])
                            break
                        else:
                            llm_need_gpus *= 2

                batch_size, expected_tpt = llm.query_candidate_tpt(
                    selected_mesh.ngpus)
                mps = llm.query_candidate_mps(selected_mesh.ngpus)

                selected_mesh.place_model(llm, mps)
                print("mesh: ", repr(selected_mesh))
                next_mesh_list.append(selected_mesh)
                find_mesh_placement = True

            if not find_mesh_placement:
                assert False, "No valid mesh size"

            mesh_list = sorted(
                next_mesh_list,
                key=lambda x:
                (sum([llm.model_size / x.ngpus
                      for llm, _ in x.models]), -x.ngpus))

        next_mesh_list = sorted(mesh_list, key=lambda x: x.ngpus, reverse=True)
        mesh_list_gpus = [x.ngpus for x in mesh_list]
        self.dump_info(next_mesh_list, info='Greedy Memory Placement')
        if dump_to_yaml:
            self.dump_to_yaml(mesh_list, mesh_list_gpus, True, dump_dir)


def build_cost_file(cost_file: str, profiling_logdir: str):
    cost_json = {}
    if os.path.exists(cost_file):
        print(f"Load cost file from {cost_file}")
        with open(cost_file, "r") as f:
            cost_json = json.load(f)

    # filename format: model_n#gpu_mps#percentage.json
    for filename in os.listdir(profiling_logdir):
        if "mps" not in filename:
            continue
        model, ngpu, mps = filename.split(".")[0].split("_")
        ngpu = str(ngpu[1:])
        mps = str(mps[3:])

        if model not in cost_json:
            cost_json[model] = {}
        if ngpu not in cost_json[model]:
            cost_json[model][ngpu] = {}
        if mps not in cost_json[model][ngpu]:
            cost_json[model][ngpu][mps] = {}

        case_cost_json = cost_json[model][ngpu][mps]
        with open(f"{profiling_logdir}/{filename}", "r") as f:
            find_result = False
            for line in f.readlines():
                if "BS, SL, Prefill Lat (ms), Decode Lat (ms)" in line:
                    find_result = True
                    continue
                if find_result:
                    split_line = line.strip().split(",")
                    if len(split_line) != 4:
                        break
                    batch_size = str(split_line[0].strip())
                    seq_len = str(split_line[1].strip())
                    if "N/A" in split_line[2]:
                        prefill_lat = -1
                    else:
                        prefill_lat = float(split_line[2].strip())
                    if "N/A" in split_line[3]:
                        decode_lat = -1
                    else:
                        decode_lat = float(split_line[3].strip())

                    if batch_size not in case_cost_json:
                        case_cost_json[batch_size] = {}
                    if seq_len not in case_cost_json[batch_size]:
                        case_cost_json[batch_size][seq_len] = {}
                    case_cost_json[batch_size][seq_len] = {
                        "prefill": prefill_lat,
                        "decoding": decode_lat,
                    }

    print(f"Save cost file to {cost_file}")
    with open(cost_file, "w") as f:
        json.dump(cost_json, f, indent=4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload-file", type=str)
    parser.add_argument("--cost-file",
                        type=str,
                        default="examples/placement/llama.json")
    parser.add_argument("--greedy", action="store_true")
    # for build cost file
    parser.add_argument("--profiling-logdir", type=str, default=None)
    args = parser.parse_args()
    return args


'''
python muxserve/muxsched/placement.py --workload-file examples/placement/models.yaml --cost-file examples/placement/llama.json
python muxserve/muxsched/placement.py --workload-file examples/placement/test.yaml --cost-file examples/placement/llama.json
'''

if __name__ == "__main__":
    args = parse_args()
    if args.profiling_logdir is not None:
        build_cost_file(args.cost_file, args.profiling_logdir)

    opt = PlacementOptimizer(args.workload_file, args.cost_file)

    dump_dir = os.path.dirname(args.workload_file) + "/yamls"
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    if args.greedy:
        opt.greedy_memory_placement(dump_dir=dump_dir)
    else:
        opt.optimize(dump_dir=dump_dir)
