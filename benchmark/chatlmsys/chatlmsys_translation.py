from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer
from typing import AsyncGenerator, List, Tuple
from typing import Dict
import yaml
import numpy as np
import copy
import json
import argparse
import random
import os

# (prompt len, output len, latency)
# REQUEST_LATENCY: List[Tuple[int, int, float]] = []
# (prompt, prompt len, output len)
REQUEST_DATA: Tuple[str, int, int]


# move the data too long and too short
# Prune too short sequences.
# This is because TGI causes errors when the input or output length
# is too short.
def filter_reqs(
        dataset_path: str,
        # num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        out_data_path: str):
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversation with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversation"]) >= 2]
    # Only keep the first two turns of each conversation.
    prompt_dataset = [(data["conversation"][0]["content"],
                       data["conversation"][1]["content"]) for data in dataset]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in prompt_dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in prompt_dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(prompt_dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append(
            (i, prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    # filtered_dataset: List[Tuple[str, int, int]] = []
    out_data = []
    for req_idx, prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            print(dataset[req_idx]["conversation_id"])
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            print(dataset[req_idx]["conversation_id"])
            continue
        # filtered_dataset.append((prompt, prompt_len, output_len))
        out_data.append(dataset[req_idx])

    with open(out_data_path, "w") as f:
        json.dump(out_data, f)


def sample_requests(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    num_requests: int = None,
    dataset: dict = None,
) -> Tuple[List[Tuple[str, int, int]], float, float]:

    if dataset is not None:
        # Load the dataset.
        with open(dataset_path) as f:
            dataset = json.load(f)

    # Filter out the conversation with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversation"]) >= 2]

    # Only keep the first two turns of each conversation.
    dataset = [(data["conversation"][0]["content"],
                data["conversation"][1]["content"]) for data in dataset]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    avg_input_len = 0
    avg_output_len = 0
    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            assert "error! should not been there"
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            assert "error! should not been there"
            continue
        # filtered_dataset.append((prompt, prompt_token_ids, prompt_len, output_len))
        filtered_dataset.append((prompt_token_ids, prompt_len, output_len))
        avg_input_len += prompt_len
        avg_output_len += output_len

    # Sample the requests.
    if num_requests is not None:
        sampled_requests = random.sample(filtered_dataset, num_requests)
    else:
        sampled_requests = filtered_dataset
    return sampled_requests, avg_input_len / len(
        sampled_requests), avg_output_len / len(sampled_requests)


def sample_arrivals(dataset: dict,
                    num_requests: int = None,
                    condense: int = 1):

    beg_tstamp = dataset[0]["tstamp"]

    dataset = [(req_idx, data["model"],
                (data["tstamp"] - beg_tstamp) / condense)
               for req_idx, data in enumerate(dataset)]

    filtered_dataset = dataset

    # Sample the requests.
    if num_requests is not None:
        sampled_requests = random.sample(filtered_dataset, num_requests)
    else:
        sample_requests = filtered_dataset

    return sample_requests


def sample_dataset(
    dataset: dict,
    num_requests: int = 1000,
):

    return {
        "arrivals": dataset["arrivals"][:num_requests],
        "requests": dataset["requests"][:num_requests],
    }


def translate(args: argparse.Namespace):
    tokenizer = get_tokenizer(args.tokenizer)

    dataset_path = args.dataset_path
    filter_out_data_path = args.filter_out_data_path
    condense = args.condense
    translated_data_path = args.translated_data_path

    if os.path.exists(filter_out_data_path):
        print("filter json exists")
    else:
        filter_reqs(dataset_path=dataset_path,
                    tokenizer=tokenizer,
                    out_data_path=filter_out_data_path)

    with open(filter_out_data_path, "r") as f:
        dataset = json.load(f)
    prompt_data, avg_input_len, avg_output_len = sample_requests(
        filter_out_data_path, tokenizer, num_requests=None, dataset=dataset)
    arrival_data = sample_arrivals(condense=condense,
                                   num_requests=None,
                                   dataset=dataset)

    assert len(prompt_data) == len(arrival_data), "error data not match"

    out_data = {
        'info': {
            'rates': [],
            'start': 0,
            'duration': 2000,
            'num_requests': len(arrival_data),
            "distribution": "chatlmsys",
            "use_share_gpt": False,
            "prompt_len": avg_input_len,
            "output_len": avg_output_len,
        },
        "arrivals": [],
        "requests": [],
    }

    llm_map = {}  # {model_name: [llm-x, beg_time, end_time, req_num, rate]}
    map_idx = 0

    for req_info, req_data in zip(arrival_data, prompt_data):
        # req_info: (req_idx, modelname, arrival_time)
        out_data["arrivals"].append(req_info[2])

        model_name = req_info[1]
        if model_name not in llm_map.keys():
            llm_map[model_name] = {
                "model_name": f'llm-{map_idx}',
                'beg_time': req_info[2],
                'end_time': 0,
                "req_num": 1,
                "rate": 0
            }
            map_idx += 1
        else:
            llm_map[model_name]['end_time'] = req_info[2]
            llm_map[model_name]['req_num'] += 1

        out_data["requests"].append({
            # "model_name": req_info[1],
            "model_name":
            llm_map[req_info[1]]['model_name'],
            "data":
            req_data,
            "slo":
            None,
            "idx":
            req_info[0],
            "time_stamp": {},
            "submit_time":
            None,
            "prefill_end_time":
            None,
            "decode_submit_time":
            None,
            "end_time":
            None,
            "is_prefill":
            True,
            "output":
            None,
            "output_idx":
            0,
            "output_tokens":
            None
        })

    for k in llm_map.keys():
        llm_map[k]['rate'] = llm_map[k]['req_num'] / (llm_map[k]['end_time'] -
                                                      llm_map[k]['beg_time'])
        out_data['info']['rates'].append(
            [llm_map[k]['model_name'], llm_map[k]['rate']])
    print(f"llm map is : {llm_map}")
    out_data['chatlmsys_map'] = llm_map

    with open(translated_data_path, "w") as f:
        json.dump(out_data, f)
        print(f'data has been dump into {translated_data_path}')


def sample_from_interval(data: Dict, start_timestamp=None, end_timestamp=None):

    ori_arrivals = data['arrivals']
    ori_reqs = data['requests']

    out_data = {'info': {}, 'arrivals': [], 'requests': []}
    out_data['info'] = copy.deepcopy(out_data['info'])

    # Determine start and end timestamps if not specified
    if start_timestamp is None:
        start_timestamp = min(ori_arrivals)
    else:
        start_timestamp += min(ori_arrivals)
    if end_timestamp is None:
        end_timestamp = max(ori_arrivals)
    else:
        end_timestamp += min(ori_arrivals)

    # add the reqs and arrivals
    llm_name_set = set()
    num_seqs, avg_in_len, avg_out_len = 0, 0, 0
    num_seqs_per_model = {}
    for arrival, req in zip(ori_arrivals, ori_reqs):
        timestamp = arrival
        if start_timestamp <= timestamp <= end_timestamp:
            input_len, output_len = req['data'][1], req['data'][2]
            avg_in_len += input_len
            avg_out_len += output_len
            num_seqs += 1
            if req['model_name'] not in num_seqs_per_model.keys():
                num_seqs_per_model[req['model_name']] = 1
            out_data['arrivals'].append(arrival - start_timestamp)
            num_seqs_per_model[req['model_name']] += 1
            out_data['requests'].append(req)
            llm_name_set.add(req['model_name'])

    # update the workload info
    out_data['info']['rates'] = []
    out_data['info']['num_requests'] = num_seqs
    out_data['info']['prompt_len'] = avg_in_len / num_seqs
    out_data['info']['output_len'] = avg_out_len / num_seqs
    for rate_info in data['info']['rates']:
        if rate_info[0] in llm_name_set:
            out_data['info']['rates'].append([
                rate_info[0], num_seqs_per_model[rate_info[0]] /
                (end_timestamp - start_timestamp)
            ])

    print(f"workload data is :{out_data['info']}")

    return out_data


def arr_rate_statistic(dataset: dict, path_workload: str):

    print(f"workload file: {path_workload}")

    # with open(path_workload) as f:
    # dataset = json.load(f)

    models_arrvals = {}

    for arrival, request in zip(dataset["arrivals"], dataset["requests"]):
        if request["model_name"] not in list(models_arrvals.keys()):
            models_arrvals[request["model_name"]] = []
        models_arrvals[request["model_name"]].append(arrival)

    ret = {}
    for model, arrivals in models_arrvals.items():
        print(
            f"model name: {model}, arrivals rate: {len(arrivals) / (max(arrivals) - min(arrivals))} req/s "
        )
        ret[model] = len(arrivals) / (max(arrivals) - min(arrivals))

    arrivals = dataset["arrivals"]

    dataset["info"] = {
        "rates": [[model, rate] for model, rate in ret.items()],
        "start": 0,
        "duration": int(max(arrivals) - min(arrivals)),
        "num_requests": len(arrivals),
        "distribution": "chatlmsys",
        "use_share_gpt": False,
        "prompt_len": 0,
        "output_len": 0
    }

    ret["total"] = len(arrivals) / (max(arrivals) - min(arrivals))
    print(
        f"total arrival rate: {len(arrivals) / (max(arrivals) - min(arrivals))} req/s"
    )

    with open(out_sample_path, "w") as f:
        json.dump(dataset, f)

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translation.")
    parser.add_argument("--tokenizer",
                        type=str,
                        default="/mnt/afs/share/LLMCKPTs/huggyllama/llama-7b",
                        help="Name or path of the tokenizer.")
    parser.add_argument("--dataset_path",
                        type=str,
                        default="/home/lurunyu/data/tinylmsys.json")
    parser.add_argument(
        "--filter_out_data_path",
        type=str,
        default=
        "/mnt/afs/lurunyu/projects/profiling-muxserve/chatlmsys/tinylmsys_filtered_first_two_round.json"
    )
    parser.add_argument(
        "--translated_data_path",
        type=str,
        default=
        "/mnt/afs/lurunyu/projects/profiling-muxserve/chatlmsys/tinylmsys_filtered_first_two_round_tranlated.json"
    )
    # parser.add_argument("--condense", type=int, default=24)
    # parser.add_argument("--sample", type=int, default=2000)
    args = parser.parse_args()
    '''
    command: python benchmark/chatlmsys/chatlmsys_translation.py
    '''

    # for chatlmsys
    args.dataset_path = "/home/lurunyu/data/chatlmsys.json"
    args.filter_out_data_path = "/home/lurunyu/data/chatlmsys_filtered_first_two_round.json"

    args.condense = 1
    args.translated_data_path = f"/home/lurunyu/data/chatlmsys_filtered_first_two_round_tranlated.json"
    args.translated_data_path = args.translated_data_path[:
                                                          -5] + f"_condense_{args.condense}" + ".json"

    # translate into the formal workload format(with condense rate)
    # translate(args)

    # file = args.translated_data_path
    # with open(file, 'r') as f:
    #     js = json.load(f)
    # print(js['info'], js['chatlmsys_map'])

    with open(args.translated_data_path, 'r') as f:
        data = json.load(f)
    time_interval = 3600 * 2
    begs, ends = [], []
    # begs.append(30 * 3600 * 24)
    # ends.append(35 * 3600 * 24)
    # begs.append(60 * 3600 * 24)
    # ends.append(65 * 3600 * 24)
    # begs.append(90 * 3600 * 24)
    # ends.append(95 * 3600 * 24)
    # begs.append(100 * 3600 * 24)
    # ends.append(105 * 3600 * 24)

    # begs.append(70 * 3600 * 24)
    # ends.append(72 * 3600 * 24)
    begs.append(55 * 3600 * 24)
    ends.append(55.5 * 3600 * 24)
    for beg, end in zip(begs, ends):
        out_data = sample_from_interval(data, beg, end)
        out_json = f'/home/lurunyu/data/requests_over_time_models_days_from_day{int(beg/3600/24)}_to_day{int(end/3600/24)}.json'
        with open(out_json, 'w') as f:
            json.dump(out_data, f)
            print(f'data has been saved to: {out_json}')

        with open('examples/chatlmsys_info.json', 'r') as f:
            lmsys_info = json.load(f)
            lmsys_map = lmsys_info['chatlmsys_map']

        name_map = {}
        for k, v in lmsys_map.items():
            name_map[v['model_name']] = v['model_type']

        # for condense in [1000, 2000, 5000, 8000, 10000]:
        # for condense in [200, 500, 1000, 2000, 3000, 4000, 5000, 8000]:
        for condense in [500, 800, 1200, 1600, 2000]:

            yaml_gen = {
                'cluster': {
                    'nnodes': 4,
                    'ngpus_per_node': 8
                },
                'models': [],
                'avg_output_len': out_data['info']['output_len'],
                'avg_prompt_len': out_data['info']['prompt_len']
            }

            condense_data = copy.deepcopy(out_data)
            arrivals = np.array(out_data['arrivals']) / condense
            condense_data['arrivals'] = arrivals.tolist()
            avg_rate, max_rate = 0, 0
            for rate_info in condense_data['info']['rates']:

                if rate_info[0] == 'llm-0':
                    continue

                rate_info[1] *= condense
                max_rate = max(rate_info[1], max_rate)
                avg_rate += rate_info[1]

                yaml_gen['models'].append({
                    'name':
                    rate_info[0],
                    'model':
                    '/mnt/afs/share/LLMCKPTs/huggyllama/llama-' +
                    name_map[rate_info[0]],
                    'rate':
                    rate_info[1]
                })

            print(f"total rate: {avg_rate}")
            avg_rate /= len(condense_data["info"]["rates"])

            condense_out_json = f'/home/lurunyu/data/requests_over_time_models_days_from_day{int(beg/3600/24)}_to_day{int(end/3600/24)}_condense{condense}_N{len(condense_data["info"]["rates"])}_maxrate_{int(max_rate)}_avgrate_{int(avg_rate)}.json'
            with open(condense_out_json, 'w') as f:
                json.dump(condense_data, f)
                print(f'data has been saved to: {condense_out_json}')

            yaml_file_path = f'benchmark/chatlmsys/yamls/requests_over_time_models_days_from_day{int(beg/3600/24)}_to_day{int(end/3600/24)}_condense{condense}_N{len(condense_data["info"]["rates"])}_maxrate_{int(max_rate)}_avgrate_{int(avg_rate)}.yaml'
            with open(yaml_file_path, 'w') as file:
                yaml.dump(yaml_gen, file, sort_keys=False)
                print(f'yaml saved to {yaml_file_path}')
