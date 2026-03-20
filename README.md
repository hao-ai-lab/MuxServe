# This Repo Will Be Maintained Under [hao-ai-lab](https://github.com/hao-ai-lab/MuxServe)

# MuxServe: Flexible Spatial-Temporal Multiplexing for LLM Serving [[paper](https://arxiv.org/abs/2404.02015)]

MuxServe is an efficient multiple LLMs serving system with flexible spatial-temporal multiplexing.

MuxServe colocates LLMs considering their popularity to multiplex memory resources, and disaggragates and flexibly colocate prefill and decoding phases leveraging their characteristics to multiplex computation resources.


## Motivation
Recent years, Large language models (LLMs) have demonstrated remarkable performance, and organizations are racing to serve LLMs of varying sizes as endpoints for use-cases like chat, programming and search. Efficienly serving multiple LLMs poses significant challenges due to varying sizes and popularity of LLMs.

MuxServe aims to serve multiple LLMs efficiently with flexible spatial-temporal multiplexing. The key insight behind is to collocate LLMs considering their popularity to multiplex memory resources, and disaggragate and flexibly colocate prefill and decoding phases leveraging their characteristics to multiplex computation resources.

## Installation

```bash
# 配置容器并安装conda
sudo docker run -dit --gpus all --ipc=host --network=host \
    -v `pwd`/data/lrc/workspace:/workspace/ \
    -v `pwd`/data/lrc/cache/huggingface/:/root/.cache/huggingface \
    --name muxserve1 \
    nvidia/cuda:11.7.0-devel-ubuntu20.04 /bin/bash

apt update && apt install -y wget bzip2
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
echo "export PATH=\$HOME/miniconda/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc
conda init bash
```

#### Prerequisites
MuxServe uses [vLLM](https://github.com/vllm-project/vllm) as the default inference engine. Please follow the instructions to install our [modified MuxServe-vLLM](https://github.com/EfficientLLMSys/MuxServe-vLLM) from source:

```bash
conda create -n muxserve python=3.9
conda activate muxserve
git clone https://github.com/EfficientLLMSys/MuxServe-vLLM.git
cd MuxServe-vLLM
pip install -e .  # This may take 5-10 minutes.
```

#### Install MuxServe from source

```bash
git clone https://github.com/EfficientLLMSys/MuxServe.git
cd MuxServe
pip install --no-build-isolation -e .
pip install -r requirements.txt
```

## Getting Start

We get start with a simple example for offline serving multiple LLMs with MuxServe ([`examples/basic`](./examples/basic/)).

#### Model config setting preparation

We've set the config file in `examples/basic/model_config.yaml`. you sould change the model checkpoint path inside the file, change `/mnt/afs/share/LLMCKPTs/huggyllama/llama-30b` into `yourpath/to/llama-30b`


#### Workload generation

We sample a workload from the [ShareGPT_V3](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json) dataset according to the workloads defined in `examples/basic/models.yaml`. The `rate` represents the arrival rate for a model in `req/s`. We can get the workload `examples/workloads/sharedgpt_n3_rate_12_5_3.json` with the following command:

```bash
python muxserve/muxsched/workload_utils.py \
    --dataset-source examples/ShareGPT_V3_unfiltered_cleaned_split.json \
    --model-yaml examples/basic/models.yaml \
    --output-file examples/basic/sharedgpt_n3_rate_10.json
```

```bash
python muxserve/muxsched/workload_utils.py \
    --model-yaml examples/basic/models.yaml \
    --trace examples/arena_trace.json \
    --output-file examples/basic/trace_rate_2.json
```

#### Set the MPS

MuxServe uses NVIDIA MPS to manage the SM resources. We can start the MPS service with the following command:

```bash
sudo bash scripts/start_mps.sh examples/basic/mps

sudo bash scripts/stop_mps.sh examples/basic/mps
```

#### Run the tests

```bash
# Modify workload-file to set rate and dataset
mkdir -p log/vllm_proc
CUDA_VISIBLE_DEVICES=0 python -m muxserve.launch examples/basic/configs/tp1_config.yaml \
    --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 \
    --nproc_per_node=1 \
    --server-port 4145 --flexstore-port 50025 \
    --workload-file examples/basic/traces/arena-trace/trace_rate_1.json \
    2>&1 | tee log/muxserve_test.log

mkdir -p log/vllm_proc
CUDA_VISIBLE_DEVICES=0,1 python -m muxserve.launch examples/basic/configs/tp2_config.yaml \
    --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 \
    --nproc_per_node=2 \
    --server-port 4145 --flexstore-port 50025 \
    --workload-file examples/basic/traces/sharedgpt/sharedgpt_n3_rate_2.json \
    2>&1 | tee log/muxserve_test.log

mkdir -p log/vllm_proc
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m muxserve.launch examples/basic/configs/tp4_config.yaml \
    --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 \
    --nproc_per_node=4 \
    --server-port 4145 --flexstore-port 50025 \
    --workload-file examples/basic/traces/sharedgpt/sharedgpt_n3_rate_2.json \
    2>&1 | tee log/muxserve_test.log
```

After starting the MPS service, we can find `nvidia-log` and `nvidia-mps` directories in `examples/basic/mps`.

#### Run MuxServe

```shell
mkdir -p log/vllm_proc
CUDA_VISIBLE_DEVICES=0 python -m muxserve.launch examples/basic/model_config.yaml \
    --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 \
    --nproc_per_node=1 \
    --server-port 4145 --flexstore-port 50025 \
    --workload-file examples/basic/trace_rate_5.json \
    2>&1 | tee log/muxserve_test.log

mkdir -p log/vllm_proc
CUDA_VISIBLE_DEVICES=1 python -m muxserve.launch examples/basic/model_config.yaml \
    --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 \
    --nproc_per_node=1 \
    --server-port 4145 --flexstore-port 50025 \
    --workload-file examples/basic/sharedgpt_n3_rate_2.json \
    2>&1 | tee log/muxserve_test.log
```

#### Run Temporal Multiplexing

close the MPS Service, see [Stop the MPS](#stop-the-mps).
Then run same command as [Run MuxServe](#run-MuxServe)

#### Run Spatial Partitioning

close the MPS Service, see [Stop the MPS](#stop-the-mps).

```shell
CUDA_VISIBLE_DEVICES=0 python -m muxserve.launch examples/basic/model_config_spatial_0.yaml \
    --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 \
    --nproc_per_node=1 \
    --server-port 4145 --flexstore-port 50025 \
    --workload-file examples/basic/sharedgpt_n3_rate_12_5_3.json \
    --split-by-model llm-0 \
    2>&1 | tee log/muxserve_test_spatial_0.log & \
CUDA_VISIBLE_DEVICES=1 python -m muxserve.launch examples/basic/model_config_spatial_1.yaml \
    --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 \
    --nproc_per_node=1 \
    --server-port 4245 --flexstore-port 51025 \
    --workload-file examples/basic/sharedgpt_n3_rate_12_5_3.json \
    --split-by-model llm-1 \
    2>&1 | tee log/muxserve_test_spatial_1.log & \
CUDA_VISIBLE_DEVICES=2,3 python -m muxserve.launch examples/basic/model_config_spatial_2.yaml \
    --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 \
    --nproc_per_node=2 \
    --server-port 4345 --flexstore-port 52025 \
    --workload-file examples/basic/sharedgpt_n3_rate_12_5_3.json \
    --split-by-model llm-2 \
    2>&1 | tee log/muxserve_test_spatial_2.log
```

#### Stop the MPS

Stop the NVIDIA MPS Service

```bash
sudo bash scripts/stop_mps.sh examples/basic/mps
```

## End-to-End Evaluations
- [Synthetic Workloads](./benchmark/end_to_end/)
- [Real Workloads](./benchmark/chatlmsys/)


## TODO
- [ ] Add support for api serve.

## Citation
```
@inproceedings{duanmuxserve,
  title={MuxServe: Flexible Spatial-Temporal Multiplexing for Multiple LLM Serving},
  author={Duan, Jiangfei and Lu, Runyu and Duanmu, Haojie and Li, Xiuhong and Zhang, Xingcheng and Lin, Dahua and Stoica, Ion and Zhang, Hao},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
