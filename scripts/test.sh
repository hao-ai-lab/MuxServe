export PYTHONPATH=$(pwd):$PYTHONPATH

# start muxserve
echo "Starting muxserve..."
python -m muxserve.launch examples/workloads/cfg_muxserve_n1_s1.yaml \
    --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 \
    --nproc_per_node=1 \
    --server-port 4145 --flexstore-port 50025 \
    --mps-dir /mnt/afs/jfduan/LLMInfer/MuxServe/log/mps \
    --workload-file examples/workloads/sharegpt_n1_rate10.json \
    2>&1 | tee log/muxserve_test.log
