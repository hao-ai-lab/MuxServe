#!/bin/bash
dir=$(dirname $0)
workdir=$(realpath $dir/..)

export PYTHONPATH=$(pwd):$PYTHONPATH

if [ "$#" -ne 3 ] && [ "$#" -ne 4 ] && [ "$#" -ne 5 ]; then
    echo "Usage: $0 <launch_type> <cuda_device> <yaml> <workload> [split_llm if 'spatial']"
    echo "bash scripts/bench_end_to_end_local.sh muxserve 0,1,2,3 ./benchmark/models_GPUnum16_mesh_size4_idx0.yaml ./benchmark/sharegpt_n10_req5081.jsonl"
    echo "bash scripts/bench_end_to_end_local.sh muxserve 0,1 ./benchmark/spatial_cfg.yaml ./benchmark/sharegpt_n10_req5081.jsonl"
    exit 1
fi

launch_type="$1"
CUDA_DEVICE="$2"
YAML="$3"
workload_file="$4"
split_llm="$5"

end_to_end_log_dir="$workdir/log/rerun/end_to_end"

flex_port_id=25
flex_server_port=48${flex_port_id}2
flex_store_port=58${flex_port_id}1
found_available_port=false
while [ "$found_available_port" = false ]; do
    result=$(netstat -tuln | grep ":$port ")
    if [ -z "$result" ]; then
        found_available_port=true
        echo "flex_server_port:   $flex_server_port avaliable"
    else
        echo "flex_server_port:   $flex_server_port not avaliable"
        ((flex_server_port++))
    fi
done
found_available_port=false
while [ "$found_available_port" = false ]; do
    result=$(netstat -tuln | grep ":$port ")
    if [ -z "$result" ]; then
        found_available_port=true
        echo "flex_store_port:      $flex_store_port avaliable"
    else
        echo "flex_store_port:      $flex_store_port not avaliable"
        ((flex_store_port++))
    fi
done

NPROC_PER_NODE=$(echo "${CUDA_DEVICE//,/ }" | wc -w)

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

ts=$(date +"%m%d%H%M")
VLLM_PROC_LOG="$end_to_end_log_dir/vllm_proc/$ts"
idx=$(echo "$YAML" | sed 's/.*idx\([0-9]\+\).*/\1/')

echo "============================================="
echo "CUDA DEVICES: ${CUDA_DEVICE}"
echo "flex_port_id: ${flex_port_id}"
echo "YAML:         ${YAML}"
echo "workload:     ${workload_file}"
echo "split_llm:    ${split_llm}"
echo "============================================="


# export CUDA_PATH=${CUDA_PATH} && \
# export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${CUDA_PATH}/extras/CUPTI/lib64:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH && \
# export PATH=/mnt/afs/dmhj/miniconda3/envs/muxserve/bin:/mnt/afs/dmhj/miniconda3/condabin:$PATH && \
# export PATH=${CUDA_PATH}/bin:$PATH && \
# export PYTHONPATH=${workspace}:${PYTHONPATH} && \

# cluster=2


model_cfg=$(realpath $YAML)
alpha=$(echo "$model_cfg" | sed -n 's/.*alpha\([0-9.]*\).*/\1/p')
scale=$(echo "$model_cfg" | sed -n 's/.*scale\([0-9.]*\).*/\1/p')
if [ $launch_type = "spatial" ]; then
    spatial_log=${end_to_end_log_dir}/spatial/alpha${alpha}_${scale}_${split_llm}.log
    mkdir -p $(dirname $spatial_log)

    VLLM_PROC_LOG=$VLLM_PROC_LOG && \
    python -m muxserve.launch $model_cfg \
                --workload-file $workload_file \
                --nproc_per_node=${NPROC_PER_NODE} \
                --server-port $flex_server_port --flexstore-port $flex_store_port \
                --split-by-model $split_llm \
                2>&1 | tee $spatial_log && \
    kill -9 $(pgrep -f $flex_store_port) \

elif [ $launch_type = "muxserve" ]; then
    # MuxServe multiplexing
    mesh_size=$(echo "$model_cfg" | sed 's/.*mesh_size\([0-9]\+\).*/\1/')
    muxserve_log=${end_to_end_log_dir}/muxserve/alpha${alpha}_${scale}_mesh${mesh_size}idx${idx}.log
    mps_log_dir=${end_to_end_log_dir}/mps/${ts}
    mkdir -p $mps_log_dir
    mkdir -p $(dirname $muxserve_log)

    export CUDA_MPS_PIPE_DIRECTORY=${mps_log_dir}/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=${mps_log_dir}/nvidia-log
    export FLEXSM_SHM_PREFIX="FLEXSM${port_id}"

    echo "MuxServe multiplexing with ${workload_file}, log to $muxserve_log"
    echo "dmhj@123" | sudo -S sh scripts/start_mps.sh $mps_log_dir
    VLLM_PROC_LOG=$VLLM_PROC_LOG && \
    python -m muxserve.launch ${model_cfg} \
        --workload-file ${workload_file} \
        --nproc_per_node=${NPROC_PER_NODE} \
        --server-port $flex_server_port --flexstore-port $flex_store_port \
        --mps-dir ${mps_log_dir} \
        --max-num-batched-tokens 2048 \
        --schedule-approach adbs \
        2>&1 | tee $muxserve_log && \
    kill -9 $(pgrep -f $flex_store_port) && \
    echo "dmhj@123" | sudo -S sh scripts/stop_mps.sh $mps_log_dir
elif [ $launch_type = "temporal" ]; then
    # temporal multiplexing
    mesh_size=$(echo "$model_cfg" | sed 's/.*mesh_size\([0-9]\+\).*/\1/')
    muxserve_log=${end_to_end_log_dir}/temporal_alpha${alpha}_${scale}_mesh${mesh_size}idx${idx}.log

    export FLEXSM_SHM_PREFIX="FLEXSM${port_id}"

    echo "MuxServe multiplexing with ${workload_file}, log to $muxserve_log"
    VLLM_PROC_LOG=$VLLM_PROC_LOG && \
    python -m muxserve.launch ${model_cfg} \
        --workload-file ${workload_file} \
        --nproc_per_node=${NPROC_PER_NODE} \
        --server-port $flex_server_port --flexstore-port $flex_store_port \
        --max-num-batched-tokens 2048 \
        --schedule-approach fcfs \
        2>&1 | tee $muxserve_log && \
    kill -9 $(pgrep -f $flex_store_port)
else
    echo "Launch type $launch_type invalid"
    exit 1
fi
