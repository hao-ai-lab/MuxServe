'''
125 days
vicuna-13b: 0.07082488310305475 requests per second
koala-13b: 0.012454756518872876 requests per second
llama-13b: 0.00518854163174257 requests per second
alpaca-13b: 0.008677358652768688 requests per second
chatglm-6b: 0.00900081100806861 requests per second
dolly-v2-12b: 0.004071699050781546 requests per second
oasst-pythia-12b: 0.003849628863633051 requests per second
stablelm-tuned-alpha-7b: 0.0033662125505506 requests per second
fastchat-t5-3b: 0.004079415484497 requests per second
gpt-3.5-turbo: 0.0012757441463659511 requests per second
gpt-4: 0.001240575436001576 requests per second
claude-1: 0.007769316375657057 requests per second
RWKV-4-Raven-14B: 0.0028183279373522356 requests per second
mpt-7b-chat: 0.0026906745384685377 requests per second
palm-2: 0.0009995623978653785 requests per second
vicuna-7b: 0.003759957164210538 requests per second
claude-instant-1: 0.0011009027886285056 requests per second
wizardlm-13b: 0.004484910447551143 requests per second
gpt4all-13b-snoozy: 0.003062738240568616 requests per second
guanaco-33b: 0.0038061470293072725 requests per second
vicuna-33b: 0.010986872164510633 requests per second
mpt-30b-chat: 0.0030795767934501976 requests per second
llama-2-13b-chat: 0.022256487179458495 requests per second
llama-2-7b-chat: 0.002814132320881063 requests per second
claude-2: 0.0016557412298015633 requests per second
'''
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import os

# # Load the JSON data
# import json
# with open('/home/lurunyu/data/chatlmsys.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)

# # Extract timestamp and model information
# timestamps = [entry['tstamp'] for entry in data]
# models = [entry['model'] for entry in data]

# dummy_data = []

# for model, timestamp in zip(models, timestamps):
#     dummy_data.append({"model":model, "tstamp":timestamp})

# with open('/home/lurunyu/data/chatlmsys_dummy.json', 'w', encoding='utf-8') as file:
#     json.dump(dummy_data, file)


def span_line(out_jpg=None,
              time_interval=24 * 3600,
              start_timestamp=None,
              end_timestamp=None):
    # Load the JSON data
    with open('/home/lurunyu/data/chatlmsys_dummy.json', 'r',
              encoding='utf-8') as file:
        data = json.load(file)

    # Extract timestamp and model information
    timestamps = [entry['tstamp'] for entry in data]

    # Determine start and end timestamps if not specified
    if start_timestamp is None:
        start_timestamp = min(timestamps)
    else:
        start_timestamp += min(timestamps)
    if end_timestamp is None:
        end_timestamp = max(timestamps)
    else:
        end_timestamp += min(timestamps)

    # Create a dictionary to store the number of requests in each time interval for each model
    model_requests_count = defaultdict(lambda: defaultdict(int))
    for entry in data:
        timestamp = entry['tstamp']
        if start_timestamp <= timestamp <= end_timestamp:
            date_object = datetime.utcfromtimestamp(timestamp)
            model = entry['model']
            interval_start = date_object.timestamp(
            ) // time_interval * time_interval
            model_requests_count[model][interval_start] += 1

    # Plotting the line graph for each model
    plt.figure(figsize=(12, 6))
    for model, requests_count in model_requests_count.items():
        sorted_requests_count = dict(sorted(requests_count.items()))
        interval_timestamps = list(sorted_requests_count.keys())
        request_numbers = list(sorted_requests_count.values())

        # Convert timestamps to days for x-axis labeling
        # plus = int((start_timestamp - min(timestamps)) / 3600 / 24)
        plus = 0
        interval_days = [(timestamp - start_timestamp) / time_interval + plus
                         for timestamp in interval_timestamps]

        # plt.plot(interval_days, request_numbers, marker=marker, linestyle='-', label=model)
        plt.plot(interval_days, request_numbers, linestyle='-', label=model)

    plt.xlabel(f'{time_interval / 3600} Hours')
    plt.ylabel('Number of Requests')
    # plt.title('Number of Requests over Time Intervals for Different Models')
    plt.title(f"{out_jpg[:-4]}")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to a file (e.g., PNG)
    if out_jpg:
        plt.savefig(out_jpg, bbox_inches='tight')
    else:
        plt.savefig('requests_over_time_models_days.png', bbox_inches='tight')


def plot_anonym_traffic(out_jpg=None,
                        time_interval=24 * 3600,
                        start_timestamp=None,
                        end_timestamp=None):
    # Load the JSON data
    with open('/home/lurunyu/data/chatlmsys_dummy.json', 'r',
              encoding='utf-8') as file:
        data = json.load(file)

    # Extract timestamp and model information
    timestamps = [entry['tstamp'] for entry in data]

    # Determine start and end timestamps if not specified
    if start_timestamp is None:
        start_timestamp = min(timestamps)
    else:
        start_timestamp += min(timestamps)
    if end_timestamp is None:
        end_timestamp = max(timestamps)
    else:
        end_timestamp += min(timestamps)

    # Create a dictionary to store the number of requests in each time interval for each model
    model_requests_count = defaultdict(lambda: defaultdict(int))
    for entry in data:
        timestamp = entry['tstamp']
        if start_timestamp <= timestamp <= end_timestamp:
            date_object = datetime.utcfromtimestamp(timestamp)
            model = entry['model']
            interval_start = date_object.timestamp(
            ) // time_interval * time_interval
            model_requests_count[model][interval_start] += 1

    # Plotting the line graph for each model
    plt.figure(figsize=(6, 2.5))
    plt.rcParams["font.family"] = "Calibri"
    for model, requests_count in model_requests_count.items():
        sorted_requests_count = dict(sorted(requests_count.items()))
        interval_timestamps = list(sorted_requests_count.keys())
        request_numbers = list(sorted_requests_count.values())

        # Convert timestamps to days for x-axis labeling
        # plus = int((start_timestamp - min(timestamps)) / 3600 / 24)
        plus = 0
        interval_days = [(timestamp - start_timestamp) / time_interval + plus
                         for timestamp in interval_timestamps]

        # plt.plot(interval_days, request_numbers, marker=marker, linestyle='-', label=model)
        plt.plot(interval_days, request_numbers, linestyle='-', label=model)

    plt.xlabel(f'Time (Hour)', fontsize=11)
    plt.yticks([])
    plt.ylabel('Request Arrival Rate', fontsize=11)
    # plt.title('Number of Requests over Time Intervals for Different Models')
    # plt.title(f"{out_jpg[:-4]}")
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to a file (e.g., PNG)
    if out_jpg:
        plt.savefig(out_jpg, bbox_inches='tight')
    else:
        plt.savefig('requests_over_time_models_days.png', bbox_inches='tight')


if __name__ == "__main__":

    # span_line(out_jpg=f"log/traffic/requests_over_time_models_days.png")

    # time_interval = 3600 * 2
    # beg = 10 * 3600 * 24
    # end = 20 * 3600 * 24
    # span_line(
    #     out_jpg=
    #     f"log/requests_over_time_models_days_from_day{int(beg/3600/24)}_to_day{int(end/3600/24)}_every_{int(time_interval / 3600)}_hours.png",
    #     time_interval=time_interval,
    #     start_timestamp=beg,
    #     end_timestamp=end)

    time_interval = 3600
    beg = 40 * 3600 * 24
    end = 60 * 3600 * 24
    plot_anonym_traffic(
        out_jpg=
        f"log/traffic/requests_over_time_models_days_from_day{int(beg/3600/24)}_to_day{int(end/3600/24)}_every_{int(time_interval / 3600)}_hours.pdf",
        time_interval=time_interval,
        start_timestamp=beg,
        end_timestamp=end)

    # time_interval = 3600 * 2
    # beg = 60 * 3600 * 24
    # end = 80 * 3600 * 24
    # span_line(
    #     out_jpg=
    #     f"log/requests_over_time_models_days_from_day{int(beg/3600/24)}_to_day{int(end/3600/24)}_every_{int(time_interval / 3600)}_hours.png",
    #     time_interval=time_interval,
    #     start_timestamp=beg,
    #     end_timestamp=end)
