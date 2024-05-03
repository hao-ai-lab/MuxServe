#!/bin/bash
dir=$(dirname $0)
workdir=$(realpath $dir/..)

export PYTHONPATH=$(pwd):$PYTHONPATH

if [ "$#" -ne 3 ] && [ "$#" -ne 4 ] && [ "$#" -ne 5 ]; then
    echo "Usage: $0 <cuda_device> <rate_list> <yaml> <(optional) log_dir> <(optional)if temporal>"
    echo "bash scripts/batch_muxserve.sh 0,1,2,3 1,2 examples/workloads/cfg_muxserve_n3.yaml"
    echo "bash scripts/batch_muxserve.sh 0,1,2,3 1,2,3,4,5,6,7 examples/workloads/cfg_muxserve_n3.yaml"
    echo "bash scripts/batch_muxserve.sh 4,5,6,7 3,4,5,6,7 examples/workloads/cfg_muxserve_n3.yaml log"
    echo "bash scripts/batch_muxserve.sh 4,5,6,7 1,2,3,4,5,6,7 examples/workloads/cfg_muxserve_n3.yaml log is_temporal"
    echo "bash scripts/batch_muxserve.sh 0,1,2,3 7,6,5,4,3,2,1 examples/workloads/cfg_muxserve_n3.yaml log is_temporal"
    exit 1
fi

CUDA_DEVICE="$1"
flex_port_ids="$2"
IFS=',' read -ra port_ids <<< "$flex_port_ids"
YAML="$3"
workload_file="$4"
LOGDIR=${5:-"log"}

NPROC_PER_NODE=$(echo "${CUDA_DEVICE//,/ }" | wc -w)

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

ts=$(date +"%m%d%H%M")
mesh_size=$(echo "$YAML" | sed 's/.*mesh_size\([0-9]\+\).*/\1/')
idx=$(echo "$YAML" | sed 's/.*idx\([0-9]\+\).*/\1/')
muxserve_log=${LOGDIR}/end_to_end/${ts}_mesh${mesh_size}idx${idx}.log
mps_log_dir=${LOGDIR}/end_to_end/mps/${ts}

mkdir -p $mps_log_dir

echo "============================================="
echo "CUDA DEVICES: ${CUDA_DEVICE}"
echo "Rates:        ${flex_port_ids}"
echo "YAML:         ${YAML}"
echo "workload:     ${workload_file}"
echo "muxserve_log:   ${muxserve_log}"
echo "mps_log_dir:  ${mps_log_dir}"
echo "============================================="

# Check if is_temporal is present in the parameters
if [[ "$@" == *is_temporal* ]]; then
    # Temporal multiplexing
    mkdir -p ${LOGDIR}/temporal
    for port_id in "${port_ids[@]}"; do
        export FLEXSM_SHM_PREFIX="TEMPORAL${port_id}"
        available_port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')

        workload_file=examples/workloads/sharegpt_uneven_n3_max${port_id}.json
        muxserve_log=${LOGDIR}/temporal/temporal_7b_13b_30b_bs256_rate${port_id}.log
        echo "Temporal multiplexing with ${workload_file}. log to ${muxserve_log}"
        python -m muxserve.launch ${YAML} \
            --nproc_per_node=${NPROC_PER_NODE} \
            --workload-file ${workload_file} \
            --server-port 39${port_id}4 --flexstore-port 57${port_id}5 \
            2>&1 | tee ${muxserve_log}
        echo -e "\n\n"
        kill -9 $(pgrep -f "57${port_id}5")
        sleep 3
    done
else
    # MuxServe multiplexing
    export CUDA_MPS_PIPE_DIRECTORY=${mps_log_dir}/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=${mps_log_dir}/nvidia-log
    for port_id in "${port_ids[@]}"; do
        muxserve_server_port=48${port_id}2
        flex_store_port=58${port_id}1
        found_available_port=false
        while [ "$found_available_port" = false ]; do
            result=$(netstat -tuln | grep ":$port ")
            if [ -z "$result" ]; then
                found_available_port=true
                echo "muxserve_server_port:   $muxserve_server_port avaliable"
            else
                echo "muxserve_server_port:   $muxserve_server_port not avaliable"
                ((muxserve_server_port++))
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

        export FLEXSM_SHM_PREFIX="FLEXSM${port_id}"
        # available_port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
        echo "MuxServe multiplexing with ${workload_file}, log to $muxserve_log"
        echo "dmhj@123" | sudo -S sh scripts/start_mps.sh $mps_log_dir
        python -m muxserve.launch ${YAML} \
            --nproc_per_node=${NPROC_PER_NODE} \
            --mps-dir ${mps_log_dir} \
            --workload-file ${workload_file} \
            --max-num-batched-tokens 2048 \
            --server-port $muxserve_server_port --flexstore-port $flex_store_port \
            --schedule-approach adbs \
            2>&1 | tee $muxserve_log
        echo -e "\n\n"
        kill -9 $(pgrep -f $flex_store_port)
        echo "dmhj@123" | sudo -S sh scripts/stop_mps.sh $mps_log_dir
    done
fi
