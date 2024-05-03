import os
import re
import numpy as np
import json
import glob
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

def parse_latency_string(latency_string):
    # Define a regular expression pattern to match the latency values
    pattern = r"\[(.*?)\] p99: (\d+\.\d+), p95: (\d+\.\d+), p90: (\d+\.\d+)"

    # Use the re.findall() function to extract the entire string and latency values
    match = re.findall(pattern, latency_string)

    # Check if there is a match
    if match:
        # Extract the latency values and the string inside square brackets from the match
        latency_label, p99, p95, p90 = match[0]
        # assert float(p99) < 20
        # if float(p99) > 1000:
        #     print(latency_label, p99)
        #     print(type(latency_label))
        #     print(latency_string)

        return latency_label, (float(p99), float(p95), float(p90))
    else:
        return None


def read_log(logfile):
    find_rate = False
    model_rates = {}
    model_tpts = {}
    model_latency_metrics = {}
    # requests = []
    requests = {}
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
                model_latency_metrics[model_name] = {}
            if "Summary: Throughput" in line:
                continue
            if "Throughput " in line:
                throughput = float(line.split()[-4])
                model_tpts[model_name] = throughput
            if '] p99: ' in line:
                latency_label, latency_metric = parse_latency_string(line)
                model_latency_metrics[model_name][latency_label] = latency_metric

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
                # requests.append(req)
                if model_name not in requests.keys():
                    requests[model_name] = []
                requests[model_name].append(req)

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

    return total_tpt, model_latency_metrics, requests


LOG_DIR = f"{os.path.dirname(__file__)}/bkp/log"
NEW_LOG_DIR = f"{os.path.dirname(__file__)}/log"

bench_list = {
    # (2.1, 1.5): {
    #     "log": {
    #         "muxserve": f"{LOG_DIR}/muxserve/alpha2.1_1.5_01271405",
    #         "spatial": f"{LOG_DIR}/spatial/alpha2.1_1.5_01271347",
    #         "temporal": f"{LOG_DIR}/temporal/alpha2.1_1.5_01271413",
    #     },
    # },
    ##################################################################
    (2.1, 1.25): {
        "log": {
            "muxserve": f"{LOG_DIR}/muxserve/alpha2.1_1.25_01271140",
            "spatial": f"{LOG_DIR}/spatial/alpha2.1_1.25_01271149",
            "temporal": f"{LOG_DIR}/temporal/alpha2.1_1.25_01271756",
        },
    },
    (2.1, 1): {
        "log": {
            "muxserve": f"{LOG_DIR}/muxserve/alpha2.1_1_01251619",
            "spatial": f"{LOG_DIR}/spatial/alpha2.1_1_01251517",
            "temporal": f"{LOG_DIR}/temporal/alpha2.1_1_01271817",
        },
    },
    (2.1, 0.75): {
        "log": {
            "muxserve": f"{LOG_DIR}/muxserve/alpha2.1_0.75_01271320",
            "spatial": f"{LOG_DIR}/spatial/alpha2.1_0.75_01271321",
            "temporal": f"{LOG_DIR}/temporal/alpha2.1_0.75_01271837",
        },
    },
    (2.1, 0.5): {
        "log": {
            "muxserve": f"{LOG_DIR}/muxserve/alpha2.1_0.5_01271225",
            "spatial": f"{LOG_DIR}/spatial/alpha2.1_0.5_01271251",
            "temporal": f"{LOG_DIR}/temporal/alpha2.1_0.5_01271858",
        },
    },
    ##################################################################
    (1.7, 1.25): {
        "log": {
            "muxserve": f"{LOG_DIR}/muxserve/alpha1.7_1.25_01280321",
            "spatial": f"{LOG_DIR}/spatial/alpha1.7_1.25_01280342",
            "temporal": f"{LOG_DIR}/temporal/alpha1.7_1.25_01280401",
        },
    },
    (1.7, 1): {
        "log": {
            "muxserve": f"{LOG_DIR}/muxserve/alpha1.7_1_01271058",
            "spatial": f"{LOG_DIR}/spatial/alpha1.7_1_01271058",
            "temporal": f"{LOG_DIR}/temporal/alpha1.7_1_01280423",
        },
    },
    (1.7, 0.75): {
        "log": {
            "muxserve": f"{LOG_DIR}/muxserve/alpha1.7_0.75_01280444",
            "spatial": f"{LOG_DIR}/spatial/alpha1.7_0.75_01280512",
            "temporal": f"{LOG_DIR}/temporal/alpha1.7_0.75_01280533",
        },
    },
    (1.7, 0.5): {
        "log": {
            "muxserve": f"{LOG_DIR}/muxserve/alpha1.7_0.5_01280604",
            "spatial": f"{LOG_DIR}/spatial/alpha1.7_0.5_01280645",
            "temporal": f"{LOG_DIR}/temporal/alpha1.7_0.5_01280705",
        },
    },

    ##################################################################
    (1.3, 1.25): {
        "log": {
            "muxserve": f"{LOG_DIR}/muxserve/alpha1.3_1.25_01280005",
            "spatial": f"{LOG_DIR}/spatial/alpha1.3_1.25_01280025",
            "temporal": f"{LOG_DIR}/temporal/alpha1.3_1.25_01280045",
        },
    },
    (1.3, 1): {
        "log": {
            "muxserve": f"{LOG_DIR}/muxserve/alpha1.3_1_01261250",
            "spatial": f"{LOG_DIR}/spatial/alpha1.3_1_01281550",
            "temporal": f"{LOG_DIR}/temporal/alpha1.3_1_01280104",
        },
    },
    (1.3, 0.75): {
        "log": {
            "muxserve": f"{LOG_DIR}/muxserve/alpha1.3_0.75_01280123",
            "spatial": f"{LOG_DIR}/spatial/alpha1.3_0.75_01280143",
            "temporal": f"{LOG_DIR}/temporal/alpha1.3_0.75_01280202",
        },
    },
    (1.3, 0.5): {
        "log": {
            "muxserve": f"{LOG_DIR}/muxserve/alpha1.3_0.5_01280222",
            "spatial": f"{LOG_DIR}/spatial/alpha1.3_0.5_01280242",
            "temporal": f"{LOG_DIR}/temporal/alpha1.3_0.5_01280301",
        },
    },

    ##################################################################
    # (0.9, 1.25): {
    #     "log": {
    #         "muxserve": f"{LOG_DIR}/muxserve/alpha0.9_1.25_01271919",
    #         "spatial": f"{LOG_DIR}/spatial/alpha0.9_1.25_01271936",
    #         "temporal": f"{LOG_DIR}/temporal/alpha0.9_1.25_01271953",
    #     },
    # },
    # (0.9, 1): {
    #     "log": {
    #         "muxserve": f"{LOG_DIR}/muxserve/alpha0.9_1_01261155",
    #         "spatial": f"{LOG_DIR}/spatial/alpha0.9_1_01261221",
    #         "temporal": f"{LOG_DIR}/temporal/alpha0.9_1_01282003",
    #     },
    # },
    # (0.9, 0.75): {
    #     "log": {
    #         "muxserve": f"{LOG_DIR}/muxserve/alpha0.9_0.75_01272034",
    #         "spatial": f"{LOG_DIR}/spatial/alpha0.9_0.75_01272103",
    #         "temporal": f"{LOG_DIR}/temporal/alpha0.9_0.75_01281814",
    #     },
    # },
    # (0.9, 0.5): {
    #     "log": {
    #         "muxserve": f"{LOG_DIR}/muxserve/alpha0.9_0.5_01272212",
    #         "spatial": f"{LOG_DIR}/spatial/alpha0.9_0.5_01272231",
    #         "temporal": f"{LOG_DIR}/temporal/alpha0.9_0.5_01281819",
    #     },
    # },

    (0.9, 1.25): {
        "log": {
            "muxserve": f"{NEW_LOG_DIR}/muxserve/alpha0.9_1.25",
            "spatial": f"{NEW_LOG_DIR}/spatial/alpha0.9_1.25",
            "temporal": f"{NEW_LOG_DIR}/temporal/alpha0.9_1.25",
        },
    },
    (0.9, 1): {
        "log": {
            "muxserve": f"{NEW_LOG_DIR}/muxserve/alpha0.9_1.0",
            "spatial": f"{NEW_LOG_DIR}/spatial/alpha0.9_1.0",
            "temporal": f"{NEW_LOG_DIR}/temporal/alpha0.9_1.0",
        },
    },
    (0.9, 0.75): {
        "log": {
            "muxserve": f"{NEW_LOG_DIR}/muxserve/alpha0.9_0.75",
            "spatial": f"{NEW_LOG_DIR}/spatial/alpha0.9_0.75",
            "temporal": f"{NEW_LOG_DIR}/temporal/alpha0.9_0.75",
        },
    },
    (0.9, 0.5): {
        "log": {
            "muxserve": f"{NEW_LOG_DIR}/muxserve/alpha0.9_0.5",
            "spatial": f"{NEW_LOG_DIR}/spatial/alpha0.9_0.5",
            "temporal": f"{NEW_LOG_DIR}/temporal/alpha0.9_0.5",
        },
    },

    ##################################################################
    (0.7, 1.25): {
        "log": {
            "muxserve": f"{LOG_DIR}/muxserve/alpha0.7_1.25_01271449",
            "spatial": f"{LOG_DIR}/spatial/alpha0.7_1.25_01271502",
            "temporal": f"{LOG_DIR}/temporal/alpha0.7_1.25_01271518",
        },
    },
    (0.7, 1): {
        "log": {
            "muxserve": f"{LOG_DIR}/muxserve/alpha0.7_1_01252307",
            "spatial": f"{LOG_DIR}/spatial/alpha0.7_1_01252308",
            "temporal": f"{LOG_DIR}/temporal/alpha0.7_1_01271543",
        },
    },
    (0.7, 0.75): {
        "log": {
            "muxserve": f"{LOG_DIR}/muxserve/alpha0.7_0.75_01271611",
            "spatial": f"{LOG_DIR}/spatial/alpha0.7_0.75_01271633",
            "temporal": f"{LOG_DIR}/temporal/alpha0.7_0.75_01271645",
        },
    },
    (0.7, 0.5): {
        "log": {
            "muxserve": f"{LOG_DIR}/muxserve/alpha0.7_0.5_01271702",
            "spatial": f"{LOG_DIR}/spatial/alpha0.7_0.5_01271717",
            "temporal": f"{LOG_DIR}/temporal/alpha0.7_0.5_01271733",
        },
    },

    # (0.7, 0.25): {
    #     "log": {
    #         "muxserve": f"{LOG_DIR}/muxserve/alpha0.7_0.25_01262347",
    #         "spatial": f"{LOG_DIR}/spatial/alpha0.7_0.25_01262347",
    #     },
    # },

    # (0.5, 1): {
    #     "log": {
    #         "muxserve": f"{LOG_DIR}/muxserve/alpha0.5_1_01262335",
    #         "spatial": f"{LOG_DIR}/spatial/alpha0.5_1_01271046",
    #     },
    # },
    # (0.5, 0.25): {
    #     "log": {
    #         "muxserve": f"{LOG_DIR}/muxserve/alpha0.5_0.25_01262346",
    #         "spatial": f"{LOG_DIR}/spatial/alpha0.5_0.25_01262346",
    #     },
    # },
}

def read_logs():

    latency_metrics = {}

    for bench_id, case in bench_list.items():
        alpha, scale = bench_id
        if alpha not in latency_metrics.keys():
            latency_metrics[alpha] = {}

        latency_metrics[alpha][scale] = {}

        for k, dname in case["log"].items():
            muxserve_or_other = k
            latency_metrics[alpha][scale][muxserve_or_other] = {'avg latency': (0, 0, 0), 'TTFT': (0, 0, 0), 'TPOT': (0, 0, 0)} # {'avg latency': (p99, p95, p90)}

            logs = os.listdir(dname)
            req_num = 0

            for log in logs:
                filepath = os.path.join(dname, log)
                total_tpt, model_latency_metrics, requests = read_log(filepath)

                for model_name, latency in model_latency_metrics.items():
                    for latency_label, lat_tuple in latency.items():
                        p99, p95, p90 = lat_tuple
                        total_p99, total_p95, total_p90 = latency_metrics[alpha][scale][muxserve_or_other][latency_label]
                        req_num +=  len(requests[model_name])
                        total_p99 += len(requests[model_name]) * p99
                        total_p95 += len(requests[model_name]) * p95
                        total_p90 += len(requests[model_name]) * p90

                        latency_metrics[alpha][scale][muxserve_or_other][latency_label] = (total_p99, total_p95, total_p90)

            for latency_label in ['avg latency', 'TTFT', 'TPOT']:
                total_p99, total_p95, total_p90 = latency_metrics[alpha][scale][muxserve_or_other][latency_label]
                total_p99 /= req_num
                total_p95 /= req_num
                total_p90 /= req_num
                # print(req_num)

                latency_metrics[alpha][scale][muxserve_or_other][latency_label] = (total_p99, total_p95, total_p90)

    return latency_metrics

def get_spec_lat(latency_metrics: dict, latency_label : str, metric_type: str):
    '''
    metric_type: p99: 0, p95: 1, p90: 2
    '''
    # transform the latency metrics into the format we want to reuse the api(plot_tpt_slo) we used before

    metric_type_str_to_int = {
        "p99": 0,
        "p95": 1,
        "p90": 2,
    }

    metric_type = metric_type_str_to_int[metric_type]

    res = {}
    for alpha, value1 in latency_metrics.items():
        if alpha not in res:
            res[alpha] = {"spatial": {}, "muxserve": {}, "temporal": {}}
        for scale, value2 in value1.items():
            for strategy, value3 in value2.items():
                res[alpha][strategy][scale] = value3[latency_label][metric_type]

    return res


def plot_p_lat(p_lat_infos: dict[str, dict]):
    platency = p_lat_infos.pop("metric_type")

    ncols = len(list(p_lat_infos.values())[0])
    # print(ncols)

    stat_avg = {'avg latency': {"faster_than_spatial": [], "faster_than_temporal": [], 'spatial_avg': [], 'temporal_avg': [], 'muxserve_avg': []} , 'TPOT': {"faster_than_spatial": [], "faster_than_temporal": [], 'spatial_avg': [], 'temporal_avg': [], 'muxserve_avg': []}, 'TTFT': {"faster_than_spatial": [], "faster_than_temporal": [], 'spatial_avg': [], 'temporal_avg': [], 'muxserve_avg': []}}

    # fig, axes = plt.subplots(nrows=len(p_lat_infos), ncols=ncols, figsize=(16, 5))
    fig, axes = plt.subplots(nrows=len(p_lat_infos), ncols=ncols, figsize=(16, 5))

    markersize=7
    linewidth=2.5
    labelsize=12

    # plot p latency
    for idx, (metric, p_lat_info) in enumerate(p_lat_infos.items()):
        for i, (ax, alpha) in enumerate(zip(axes[idx], p_lat_info)):
            title = f"α={alpha}"

            key = sorted(p_lat_info[alpha]["spatial"].keys())
            # print(key)

            spatial_tpt = [p_lat_info[alpha]["spatial"][k] for k in key]
            temporal_tpt = [p_lat_info[alpha]["temporal"].get(k,0) for k in key]
            muxserve_tpt = [p_lat_info[alpha]["muxserve"][k] for k in key]


            x_ticks = [x * 1/key[0] for x in key] # normalization
            ax.plot(x_ticks, spatial_tpt, label="Spatial Partitioning", marker="o", linewidth=linewidth, markersize=markersize)
            ax.plot(x_ticks, temporal_tpt, label="Temporal Multiplexing", marker="o", linewidth=linewidth, markersize=markersize)
            ax.plot(x_ticks, muxserve_tpt,  label="MuxServe",  marker="o", linewidth=linewidth, markersize=markersize)

            faster_than_spatial = [(b - a) / b for (a, b) in zip(muxserve_tpt, spatial_tpt)]
            faster_than_temporal = [(b - a) / b for (a, b) in zip(muxserve_tpt, temporal_tpt)]
            ax.plot(x_ticks, faster_than_spatial, label="faster_than_spatial Partitioning", marker="o", linewidth=linewidth, markersize=markersize)
            ax.plot(x_ticks, faster_than_temporal, label="faster_than_temporal Multiplexing", marker="o", linewidth=linewidth, markersize=markersize)

            stat_avg[metric]['faster_than_spatial'].append(np.average(faster_than_spatial))
            stat_avg[metric]['faster_than_temporal'].append(np.average(faster_than_temporal))
            stat_avg[metric]['spatial_avg'].append(np.average(spatial_tpt))
            stat_avg[metric]['temporal_avg'].append(np.average(temporal_tpt))
            stat_avg[metric]['muxserve_avg'].append(np.average(muxserve_tpt))

            ax.grid()

            # Add text annotations for each coordinate
            for x, y in zip(x_ticks, faster_than_spatial):
                ax.text(x, y, f'{y:.1f}', fontsize=8, ha='center', va='bottom', color='black')

            for x, y in zip(x_ticks, faster_than_temporal):
                ax.text(x, y, f'{y:.1f}', fontsize=8, ha='center', va='bottom', color='black')

            if i == 0:
                ax.set_ylabel(f"{metric} {platency} ($s$)", fontsize=labelsize)

            ax.set_title(title, fontsize=labelsize)
            ax.set_xlabel("Rate Scale", fontsize=labelsize)

    for metric, v1 in stat_avg.items():
        print(f"{metric}: ")
        for faster, v2 in v1.items():
            print(f"  {faster}: {np.mean(v2):.2f}")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles,
               labels,
               loc='upper center',
               ncol=6,
               bbox_to_anchor=(0.5, 1.06),
               fontsize=labelsize)

    fig.dpi=600
    plt.tight_layout()
    fig.savefig("e2e.pdf", bbox_inches='tight', pad_inches=0.05)
    # fig.savefig("e2e.jpg", bbox_inches='tight', pad_inches=0.05)

import pickle
def load_or_compute_latency_metrics(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            latency_metrics = pickle.load(f)
    else:
        latency_metrics = read_logs()
        with open(cache_file, 'wb') as f:
            pickle.dump(latency_metrics, f)
    return latency_metrics

cache_file = 'latency_metrics_cache.pkl'

if __name__ == "__main__":
    # latency_metrics = read_logs()
    latency_metrics = load_or_compute_latency_metrics(cache_file)
    print(latency_metrics)

    # metric_type: ["p99", "p95", "p90"]
    metric_type = "p99"
    res = {"metric_type": None}

    for typ in ['avg latency', 'TPOT', 'TTFT']:
        res[typ] = get_spec_lat(latency_metrics, typ, metric_type)

    print("begin plot ...")
    plot_p_lat(res)
