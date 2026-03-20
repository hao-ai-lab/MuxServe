import os
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Request:
    req_id: int
    model_name: str
    prompt_len: int
    output_len: int
    arrival_time: int
    submit_time: int
    prefill_end: int
    end: int


colors = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2"]


def read_log(logfile):
    find_rate = False
    model_rates = {}
    model_tpts = {}
    requests = []
    with open(logfile, "r") as f:
        model_name = None
        for line in f:
            if "Workload Statistics:" in line:
                find_rate = True
            if find_rate and "Model: " in line and "rate: " in line:
                model = line.split()[-3]
                rate = float(line.split()[-1])
                model_rates[model] = rate

            if "Name: " in line:
                model_name = line.split()[-1]
            if "Summary: Throughput" in line:
                continue
            if "Throughput " in line:
                throughput = float(line.split()[-4])
                model_tpts[model_name] = throughput

            if "Request" in line:
                request_info = line.strip().split()
                req_id = int(request_info[request_info.index("Request") + 1])
                model_name = request_info[request_info.index("model") + 1]
                prompt_len = int(request_info[request_info.index("prompt") +
                                              1])
                output_len = int(request_info[request_info.index("output") +
                                              1])
                arrival_time = float(
                    request_info[request_info.index("arrival") + 1])
                submit_time = float(request_info[request_info.index("submit") +
                                                 1])
                prefill_end = float(
                    request_info[request_info.index("prefill_end") + 1])
                end = float(request_info[request_info.index("end") + 1])
                req = Request(req_id, model_name, prompt_len, output_len,
                              arrival_time, submit_time, prefill_end, end)
                requests.append(req)

    total_tpt = 0
    total_rate = 0
    for model in model_rates:
        if model not in model_tpts:
            continue
        total_rate += model_rates[model]
        total_tpt += model_rates[model] * model_tpts[model]
    if total_rate == 0:
        print(logfile)
    else:
        total_tpt = total_tpt / total_rate
    return total_tpt, requests


baseline = {
    'llama-13b': {
        32: {
            'decoding': 0.0198,
            'prefill': 0.020882
        },
        64: {
            'decoding': 0.0198015,
            'prefill': 0.022158999999999998
        },
        128: {
            'decoding': 0.019932333333333333,
            'prefill': 0.029684000000000002
        },
        256: {
            'decoding': 0.020116999999999996,
            'prefill': 0.039314
        },
        512: {
            'decoding': 0.020397199999999997,
            'prefill': 0.07577500000000001
        }
    },
    'llama-30b': {
        32: {
            'decoding': 0.025072,
            'prefill': 0.036543
        },
        64: {
            'decoding': 0.0249465,
            'prefill': 0.037311
        },
        128: {
            'decoding': 0.024406666666666667,
            'prefill': 0.036682
        },
        256: {
            'decoding': 0.0241195,
            'prefill': 0.038727
        },
        512: {
            'decoding': 0.024103199999999998,
            'prefill': 0.062819
        }
    },
    'llama-65b': {
        32: {
            'decoding': 0.032904,
            'prefill': 0.049544
        },
        64: {
            'decoding': 0.0318775,
            'prefill': 0.049874
        },
        128: {
            'decoding': 0.031788,
            'prefill': 0.05021
        },
        256: {
            'decoding': 0.031942,
            'prefill': 0.054674999999999994
        },
        512: {
            'decoding': 0.031767000000000004,
            'prefill': 0.077788
        }
    },
    'llama-7b': {
        32: {
            'decoding': 0.011264,
            'prefill': 0.014082
        },
        64: {
            'decoding': 0.011244,
            'prefill': 0.014129
        },
        128: {
            'decoding': 0.011326000000000001,
            'prefill': 0.016023
        },
        256: {
            'decoding': 0.01144325,
            'prefill': 0.023747
        },
        512: {
            'decoding': 0.011651400000000001,
            'prefill': 0.041027
        }
    }
}


def get_map_llm():
    chatlmsys_info_file = "benchmark/chatlmsys/chatlmsys_info.json"
    with open(chatlmsys_info_file, 'r') as f:
        chatlmsys_info = json.load(f)

    map_llm = {}
    for v in chatlmsys_info["chatlmsys_map"].values():
        map_llm[v['model_name']] = 'llama-' + v['model_type']
    return map_llm


map_llm = get_map_llm()


def estimate_base_latency(model_type, prompt_len, out_len):
    # 1. get estimation of prefill latency
    bounds = [32, 64, 128, 256, 512]
    for i, bound in enumerate(bounds):
        if prompt_len >= 512:
            prefill_lt = baseline[model_type][512]["prefill"] * (prompt_len /
                                                                 512)
            break
        elif bounds[i + 1] >= prompt_len >= bound:
            lo = baseline[model_type][bound]["prefill"]
            hi = baseline[model_type][bounds[i + 1]]["prefill"]
            prefill_lt = ((prompt_len - bound) /
                          (bounds[i + 1] - bound)) * (hi - lo)
            break
        else:
            prefill_lt = baseline[model_type][32]["prefill"]
            break

    # 2. get estimation of decode latency
    decoding_lt = sum(x["decoding"]
                      for x in baseline[model_type].values()) / len(
                          baseline[model_type])
    return prefill_lt + out_len * decoding_lt


def compute_slo_attainment(requests, scale=5):
    within_slo = 0
    for req in requests:
        model = req.model_name
        prompt_len = req.prompt_len
        output_len = req.output_len
        request_lat = req.end - req.arrival_time
        base_lat = estimate_base_latency(map_llm[model], prompt_len,
                                         output_len)
        if request_lat <= base_lat * scale:
            within_slo += 1
    return within_slo / len(requests)


logdir = "/mnt/afs/lurunyu/projects/MuxServe/log/benchmark/chatlmsys/yamls"
alphas = [500, 800, 1200, 1600, 2000]
rates_base = [3.0, 4.8, 7.1, 9.5, 11.9]
rate_ticks = [1.0, 1.6, 2.4, 3.2, 4.0]

approaches = {
    "spatial": "Spatial",
    "temporal": "Temporal",
    "muxserve": "MuxServe",
}


def read_logs(directory, alpha):
    all_tpt = 0
    req_list = []
    for single_dir in os.listdir(directory):
        if str(alpha) in single_dir and 'day55_to_day55' in single_dir:
            tpt, requests = read_log(
                os.path.join(directory, single_dir, 'log.log'))
            req_list = req_list + requests
            all_tpt += tpt * len(requests)
    all_tpt /= len(req_list)
    return all_tpt, req_list


datas = {}
slos = {}
for approach in approaches:
    datas[approach] = []
    slos[approach] = []
    for alpha in alphas:
        logdir_ = f"{logdir}/{approach}/"
        tpt, requests = read_logs(logdir_, alpha)
        datas[approach].append(tpt)
        slo_att = compute_slo_attainment(requests, scale=8)
        slos[approach].append(slo_att * 100)

fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.2), dpi=300)
# ax[0].set_xlabel("Rate Scale")
ax[0].set_xlabel("Avg Rate ($req/s$)")
ax[0].set_ylabel("Throughput ($req/s$)")
# ax[0].set_xticks(alphas)
# ax[0].set_xticklabels(alphas)
ax[0].set_xticks(rates_base)
ax[0].set_xticklabels(rates_base)
baseline = datas["temporal"]
for i, approach in enumerate(approaches):
    # ax[0].plot(alphas, datas[approach], label=approaches[approach], marker="o")
    ax[0].plot(rates_base,
               datas[approach],
               label=approaches[approach],
               marker="o")
    speedup = [x / y for x, y in zip(datas[approach], baseline)]
    print(approach, speedup)
ax[0].grid()

# ax[1].set_xlabel("Rate Scale")
ax[1].set_xlabel("Avg Rate ($req/s$)")
ax[1].set_ylabel("SLO Attainment (%)")
# ax[1].set_xticks(alphas)
# ax[1].set_xticklabels(alphas)
ax[1].set_xticks(rates_base)
ax[1].set_xticklabels(rates_base)
for i, approach in enumerate(approaches):
    # ax[1].plot(alphas, slos[approach], label=approaches[approach], marker="o")
    ax[1].plot(rates_base,
               slos[approach],
               label=approaches[approach],
               marker="o")
ax[1].grid()

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles,
           labels,
           loc='upper center',
           ncol=4,
           bbox_to_anchor=(0.5, 1.1))
fig.tight_layout()
# fig.savefig("benchmark/chatlmsys/chatlmsys.jpg",
fig.savefig("benchmark/chatlmsys/chatlmsys.pdf",
            bbox_inches='tight',
            pad_inches=0.05)
fig.show()
