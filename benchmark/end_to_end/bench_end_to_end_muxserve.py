import os
import json
import yaml

from pprint import pprint
from typing import Dict, List

END_TO_END_DIR = os.path.dirname(__file__)
PROJ_DIR = f"{os.path.dirname(__file__)}/../.."

MODEL_TO_PATH = {
    "llama-7b": "/mnt/afs/share/LLMCKPTs/huggyllama/llama-7b",
    "llama-13b": "/mnt/afs/share/LLMCKPTs/huggyllama/llama-13b",
    "llama-30b": "/mnt/afs/share/LLMCKPTs/huggyllama/llama-30b",
    "llama-65b": "/mnt/afs/share/LLMCKPTs/huggyllama/llama-65b",
}
# path to `ShareGPT_V3_unfiltered_cleaned_split.json`
SHAREGPT_PATH = "/mnt/afs/dmhj/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
# this is for caching tokenized ShareGPT_V3 dataset. Specify it to accelerate config generation
TOKENIZED_DATA_CACHE = "/mnt/afs/dmhj/datasets/ShareGPT_V3_llama_tokenized.cache"
# statistic cost file
COST_FILE = f"{PROJ_DIR}/examples/placement/llama.json"


def gen_models_yaml(
    nnodes: int,
    ngpus_per_node: int,
    model_to_rate: Dict[str, List[float]],
    dump_path: str,
):
    data = {
        "cluster": {
            "nnodes": nnodes,
            "ngpus_per_node": ngpus_per_node
        },
        "models": [],
    }

    model_id = 0
    for model, rates in model_to_rate.items():
        for rate in rates:
            data["models"].append({
                "name": f"llm-{model_id}",
                "model": MODEL_TO_PATH[model],
                "rate": rate
            })
            model_id += 1

    with open(dump_path, "w") as fp:
        yaml.dump(data, fp, sort_keys=False)


def get_workload_file_from_yaml(models_yaml: str, dump_dir: str, **kwargs):
    from muxserve.muxsched.workload_utils import get_workloads_info_from_yaml, generate_workload

    workload_infos = get_workloads_info_from_yaml(models_yaml)
    print(f"Get workload info from {models_yaml}:\n{workload_infos}")

    rate = max(v[1] for v in workload_infos)
    num_models = len(workload_infos)
    output_file = os.path.join(dump_dir,
                               f"sharegpt_n{num_models}_rate{rate}.json")

    generate_workload(workload_infos, output_file, **kwargs)


def get_workload_from_optimized_placement(
    info: dict[str, dict],
    time: int,
    models_yaml: str,
    dump_dir: str,
    **kwargs,
):
    from muxserve.muxsched.workload_utils import get_workloads_info_from_yaml, generate_workload, sample_request_datas

    workload_infos = get_workloads_info_from_yaml(models_yaml)

    llm_tpt = []
    info.pop("muxserve_tpt")
    for mesh_id, llms in info.items():
        for llm_id, llm_info in llms.items():
            llm_tpt.append((llm_id, llm_info["expected_tpt"]))

    llm_tpt.sort()
    pprint(f"llm_tpt:\n{llm_tpt}")

    sampled_req = []
    num_req = []
    for llm_id, model_tpt in llm_tpt:
        cur_num_req = int(model_tpt * time * 1.1)
        num_req.append(cur_num_req)
        sampled_req.append(
            sample_request_datas(cur_num_req,
                                 SHAREGPT_PATH,
                                 tokenized_cache_path=TOKENIZED_DATA_CACHE))
    max_num_req = max(num_req)

    kwargs.update({
        "sampled_requests": sampled_req,
        "num_requests": num_req,
    })
    output_file = os.path.join(
        dump_dir, f"sharegpt_n{len(llm_tpt)}_req.json")

    generate_workload(workload_infos, output_file, **kwargs)


def get_placement_from_cfg(models_yaml: str,
                           costfile: str,
                           is_greedy=False,
                           dump_to_yaml=True,
                           dump_dir: str = None,
                           verbose: bool = False):
    from muxserve.muxsched.placement import PlacementOptimizer

    opt = PlacementOptimizer(models_yaml, costfile)

    return opt.optimize(is_greedy,
                        dump_dir=dump_dir,
                        dump_to_yaml=dump_to_yaml,
                        verbose=verbose)


def is_cfg_valid(read_dir: str) -> list[str]:
    res = []
    for cfg_file in os.listdir(read_dir):
        if "spatial" in cfg_file:
            continue
        if os.path.isdir(f"{read_dir}/{cfg_file}"):
            continue

        with open(f"{read_dir}/{cfg_file}", "r") as fp:
            cfg = yaml.safe_load(fp)

        if cfg["gpu_memory_utilization"] <=0:
            res.append(f"{read_dir}/{cfg_file}")

    return res


def gen_spatial_cfg_from_muxserve_cfg(read_dir: str):
    model2placement = {
        "llama-7b": [0],
        "llama-13b": [0],
        "llama-30b": [0,1,2,3],
        "llama-65b": [0,1,2,3,4,5,6,7],
    }

    models: list[dict] = []
    for cfg_file in os.listdir(read_dir):
        if "spatial" in cfg_file:
            continue

        with open(f"{read_dir}/{cfg_file}", "r") as fp:
            cfg = yaml.safe_load(fp)

        models+=cfg["models"]

    # print(models)
    # print()

    for i in range(len(models)):
        model_type = models[i]["model"].split("/")[-1]
        models[i].update({
            "placement":[[x for x in model2placement[model_type] ]],
            "tensor_parallel_size": len(model2placement[model_type]),
            "mps_percentage": [100, 90],
            "max_num_seqs": 256,
        })

    res = {
        "num_gpus": 8,
        "max_num_seqs": 256,
        "overload_threshold": 2,
        "gpu_memory_utilization": 0.5,
        "models": models,
        "workloads": {
            "workload_file": None
        },
    }

    yaml.Dumper.add_representer(type(None),lambda dumper, value: dumper.represent_scalar(u'tag:yaml.org,2002:null', ''))

    with open(f"{read_dir}/spatial_cfg.yaml", "w")as fp:
        fp.write("# 1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_4_4_8\n")
        yaml.dump(res, fp, sort_keys="False")


def gen_temporal_cfg_from_muxserve_cfg(read_dir: str):
    yaml.Dumper.add_representer(type(None),lambda dumper, value: dumper.represent_scalar(u'tag:yaml.org,2002:null', ''))
    for cfg_file in os.listdir(read_dir):
        if "spatial" in cfg_file:
            continue
        if os.path.isdir(f"{read_dir}/{cfg_file}"):
            continue
        if not cfg_file.endswith(".yaml"):
            continue

        with open(f"{read_dir}/{cfg_file}", "r") as fp:
            cfg = yaml.safe_load(fp)

        for model in cfg["models"]:
            model.update({
            "mps_percentage": [100, 90],
            # "max_num_seqs": 256,
        })

        temporal_dir = f"{read_dir}/temporal"
        if not os.path.exists(temporal_dir):
            os.makedirs(temporal_dir)

        with open(f"{temporal_dir}/{cfg_file}", "w")as fp:
            yaml.dump(cfg, fp, sort_keys="False")


def gen_spatial_cfg(muxserve_cfg_dir: str):
    for sub_cfg_dir in os.listdir(muxserve_cfg_dir):
        cfg_dir = f"{muxserve_cfg_dir}/{sub_cfg_dir}"
        if not os.path.isdir(cfg_dir):
            continue

        gen_spatial_cfg_from_muxserve_cfg(cfg_dir)


def gen_temporal_cfg(muxserve_cfg_dir: str):
    for sub_cfg_dir in os.listdir(muxserve_cfg_dir):
        cfg_dir = f"{muxserve_cfg_dir}/{sub_cfg_dir}"
        if not os.path.isdir(cfg_dir):
            continue

        gen_temporal_cfg_from_muxserve_cfg(cfg_dir)


def get_real_rate(rates_ratio: list[float], max_rate: float,
                  rate_scale: float) -> list[float]:
    scaled_max_rate = max_rate * rate_scale
    scale = scaled_max_rate / rates_ratio[0]
    real_rates = [x * scale for x in rates_ratio]

    return real_rates


def gen_power_law_dis(alpha: float, num_models: int) -> list[float]:
    rates = [(x + 1)**(-alpha) for x in range(num_models)]
    rates_sum = sum(rates)
    rates_ratio = [x / rates_sum for x in rates]

    return rates_ratio


def assign_rates(real_rates: list[float],
                 model2num: dict[str, int]) -> dict[str, list[float]]:
    assert sum(model2num.values()) == len(real_rates)
    res = {k: None for k in model2num}

    cur = 0
    for k, v in model2num.items():
        res[k] = real_rates[cur:cur + v]
        cur += v

    return res


def gen_config_with_power_law(config_dir: str, workloads_dir: str):
    num_models = 19  # 12 x 7B; 4 x 13B; 2 x 30B; 1 x 65B
    alpha_lis = [0.7, 0.9, 1.3, 1.7, 2.1]
    max_rate_lis = [40]
    rate_scale_lis = [0.5, 0.75, 1.0, 1.25] # 20, 30, 40, 50
    model2num = {
        "llama-7b": 12,
        "llama-13b": 4,
        "llama-30b": 2,
        "llama-65b": 1,
    }
    nnodes = 4
    ngpus_per_node = 8
    tmp_cfg = "/tmp/tmp_model_cfg.yaml"

    if not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)

    flog = open(f"{config_dir}/gen_pl.log", "w")

    for alpha in alpha_lis:
        print(f"* α: {alpha}")
        for rate_scale in rate_scale_lis:
            rates_ratio = gen_power_law_dis(alpha, num_models)
            print(f"=== rate scale: {rate_scale}")
            for max_rate in max_rate_lis:
                print(f">>> Max Rate: {max_rate}")

                cfg_dir = f"{config_dir}/alpha{alpha}_scale{rate_scale}_max{max_rate}"
                if not os.path.exists(cfg_dir):
                    os.makedirs(cfg_dir, exist_ok=True)

                real_rate = get_real_rate(rates_ratio, max_rate, rate_scale)
                rate_map = assign_rates(real_rate, model2num)

                gen_models_yaml(nnodes, ngpus_per_node, rate_map, tmp_cfg)

                muxserve_placement = get_placement_from_cfg(tmp_cfg,
                                                    COST_FILE,
                                                    dump_to_yaml=True,
                                                    dump_dir=cfg_dir,
                                                    verbose=False)

                gen_spatial_cfg_from_muxserve_cfg(cfg_dir)
                gen_temporal_cfg_from_muxserve_cfg(cfg_dir)

                workloads_dump_dir = f"{workloads_dir}/alpha{alpha}_scale{rate_scale}_max{max_rate}"
                if not os.path.exists(workloads_dump_dir):
                    os.makedirs(workloads_dump_dir, exist_ok=True)

                workload_args = {
                    "start": 0,
                    "duration": 1000,
                    "distribution": "poisson",
                    "prompt_distribution": None,
                    "use_share_gpt": True,
                    "prompt_len": None,
                    "output_len": None,
                    "dataset": SHAREGPT_PATH,
                }
                get_workload_from_optimized_placement(
                    muxserve_placement,
                    time=240,
                    models_yaml=tmp_cfg,
                    dump_dir=workloads_dump_dir,
                    **workload_args)

                flog.write(f"{cfg_dir}\n{json.dumps(muxserve_placement)}\n")

    flog.close()


if __name__ == "__main__":
    muxserve_cfg_dir = f"{END_TO_END_DIR}/model_cfgs"
    workloads_dir = f"{END_TO_END_DIR}/workloads"
    models_yaml_path = f"{END_TO_END_DIR}/models.yaml"

    gen_config_with_power_law(muxserve_cfg_dir, workloads_dir)
