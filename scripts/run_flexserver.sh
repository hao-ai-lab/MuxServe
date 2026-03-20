#!/bin/bash

###########################################################
# for single card test without zmq server
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    muxserve/flexserver/muxserve_server.py \
    --model /mnt/afs/share/LLMCKPTs/huggyllama/llama-7b \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --block-size 16 \
    --swap-space 1 \
    --load-format dummy \
    --workload-file /mnt/afs/lurunyu/projects/profiling-muxserve/workloads/workload_bs1_inputlen64_outputlen64.json \
    --mps-percentage 40
###########################################################


###########################################################
# for single card test with zmq server
python -m muxserve.entrypoint --flexstore --model-config examples/test_cfg.yaml \
    --mps-dir /home/lurunyu/projects/profilig-muxserve/log/mps1 \
    --gpu-memory-utilization 0.2 \
    --flexstore-port 51051

torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    muxserve/flexserver/muxserve_server.py \
    --model /mnt/afs/share/LLMCKPTs/huggyllama/llama-7b \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --block-size 16 \
    --swap-space 1 \
    --load-format dummy \
    --flexstore-port 51051 \
    --workload-file /mnt/afs/lurunyu/projects/profiling-muxserve/workloads/workload_bs1_inputlen64_outputlen64.json \
    --mps-percentage 40
###########################################################


###########################################################
# for multi-card test(not support yet)
python -m muxserve.entrypoint --flexstore --model-config examples/model_cfg.yaml \
    --mps-dir /home/lurunyu/projects/profilig-muxserve/log/mps1 \
    --gpu-memory-utilization 0.2 \
    --flexstore-port 50051

torchrun --standalone --nnodes=1 --nproc-per-node=4 \
    muxserve/flexserver/muxserve_server.py \
    --model /mnt/afs/share/LLMCKPTs/huggyllama/llama-7b \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 4 \
    --block-size 16 \
    --swap-space 1 \
    --load-format dummy \
    --flexstore-port 50051 \
    --workload-file /mnt/afs/lurunyu/projects/profiling-muxserve/workloads/workload_bs1_inputlen64_outputlen64.json \
    --mps-percentage 40
###########################################################
