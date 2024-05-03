# MuxServe: Flexible Spatial-Temporal Multiplexing for LLM Serving [[paper](https://arxiv.org/abs/2404.02015)]

MuxServe is an efficient multiple LLMs serving system with flexible spatial-temporal multiplexing.

MuxServe colocates LLMs considering their popularity to multiplex memory resources, and disaggragates and flexibly colocate prefill and decoding phases leveraging their characteristics to multiplex computation resources.


## Motivation
Recent years, Large language models (LLMs) have demonstrated remarkable performance, and organizations are racing to serve LLMs of varying sizes as endpoints for use-cases like chat, programming and search. Efficienly serving multiple LLMs poses significant challenges due to varying sizes and popularity of LLMs.

MuxServe aims to serve multiple LLMs efficiently with flexible spatial-temporal multiplexing. The key insight behind is to collocate LLMs considering their popularity to multiplex memory resources, and disaggragate and flexibly colocate prefill and decoding phases leveraging their characteristics to multiplex computation resources.

## Installation

#### Prerequisites
MuxServe uses [vLLM](https://github.com/vllm-project/vllm) as the default inference engine. Please follow the instructions to install our [modified MuxServe-vLLM](https://github.com/EfficientLLMSys/MuxServe-vLLM) from source:


```
conda create -n muxserve python=3.9
conda activate muxserve
git clone https://github.com/EfficientLLMSys/MuxServe-vLLM.git
cd MuxServe-vLLM
pip install -e .  # This may take 5-10 minutes.
```

#### Install MuxServe from source

```
git clone https://github.com/EfficientLLMSys/MuxServe.git
cd MuxServe
pip install -e .
pip install -r requirements.txt
```

## Getting Start

We get start with a simple example for offline serving multiple LLMs with MuxServe ([`examples/basic`](./examples/basic/)).

#### Model config setting preparation

We've set the config file in `examples/basic/model_config.yaml`. you sould change the model checkpoint path inside the file, change `/mnt/afs/share/LLMCKPTs/huggyllama/llama-30b` into `yourpath/to/llama-30b`


#### Workload generation

We sample a workload from the [ShareGPT_V3](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json) dataset according to the workloads defined in `examples/basic/models.yaml`. The `rate` represents the arrival rate for a model in `req/s`. We can get the workload `examples/workloads/sharedgpt_n3_rate_12_5_3.json` with the following command:

```shell
python muxserve/muxsched/workload_utils.py \
    --dataset-source /yourpathto/ShareGPT_V3_unfiltered_cleaned_split.json \
    --workload_info_from_yaml True \
    --output-file examples/basic/sharedgpt_n3_rate_12_5_3.json \
    --model-yaml examples/basic/models.yaml
```

#### Set the MPS

MuxServe uses NVIDIA MPS to manage the SM resources. We can start the MPS service with the following command:

```shell
sudo bash scripts/start_mps.sh examples/basic/mps
```

After starting the MPS service, we can find `nvidia-log` and `nvidia-mps` directories in `examples/basic/mps`.

#### Run MuxServe

```shell
mkdir -p log/vllm_proc
python -m muxserve.launch examples/basic/model_config.yaml \
    --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 \
    --nproc_per_node=4 \
    --server-port 4145 --flexstore-port 50025 \
    --workload-file examples/basic/sharedgpt_n3_rate_12_5_3.json \
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
@article{duan2024muxserve,
  title={MuxServe: Flexible Multiplexing for Efficient Multiple LLM Serving},
  author={Duan, Jiangfei and Lu, Runyu and Duanmu, Haojie and Li, Xiuhong and Zhang, Xingcheng and Lin, Dahua and Stoica, Ion and Zhang, Hao},
  journal={arXiv preprint arXiv:2404.02015},
  year={2024}
}
```
