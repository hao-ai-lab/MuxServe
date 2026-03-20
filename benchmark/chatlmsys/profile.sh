#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH


if [ "$#" -ne 6 ]; then
    echo "Usage: $0 nnodes nprocs yaml mpsfile workloads cuda_devices"
    echo "sudo bash benchmark/chatlmsys/profile.sh 1 2 benchmark/chatlmsys/yamls/placement_gen/requests_over_time_models_days_from_day60_to_day65_condense500_N19_maxrate_7_avgrate_1_GPUnum32_mesh_size2_idx8.yaml /mnt/afs/lurunyu/projects/MuxServe/log/mps1 /mnt/afs/lurunyu/data/requests_over_time_models_days_from_day60_to_day65_condense500_N19_maxrate_7_avgrate_1.json 0,1"
    echo "sudo bash benchmark/chatlmsys/profile.sh 1 4 benchmark/chatlmsys/yamls/placement_gen/requests_over_time_models_days_from_day30_to_day35_condense1000_N14_maxrate_19_avgrate_6_GPUnum32_mesh_size4_idx2.yaml /mnt/afs/lurunyu/projects/MuxServe/log/mps1 /mnt/afs/lurunyu/data/requests_over_time_models_days_from_day30_to_day35_condense1000_N14_maxrate_19_avgrate_6.json 4,5,6,7"
    echo "sudo bash benchmark/chatlmsys/profile.sh 1 4 benchmark/chatlmsys/yamls/placement_gen/requests_over_time_models_days_from_day60_to_day65_condense800_N19_maxrate_11_avgrate_2_GPUnum32_mesh_size2_idx7.yaml /mnt/afs/lurunyu/projects/MuxServe/log/mps1 /mnt/afs/lurunyu/data/requests_over_time_models_days_from_day60_to_day65_condense800_N19_maxrate_11_avgrate_2_GPUnum32_mesh_size2_idx7.json 2,3"
    exit 1
fi

get_available_port() {
    local port
    port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
    echo "$port"
}

echo "You should begin to open MPS $MPS_FILE First!!"
echo "You should begin to open MPS $MPS_FILE First!!"
echo "You should begin to open MPS $MPS_FILE First!!"
echo "sudo bash scripts/start_mps.sh $MPS_FILE"

NNODES="$1"
NPROCS="$2"
YAML="$3"
MPS_FILE=${4:-"/mnt/afs/lurunyu/projects/MuxServe/log/mps"}
workload_file="$5"
# IFS=',' read -ra scales <<< "$6"
CUDA_DEVICE="$6"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

directory=$(dirname "$YAML")
filename=$(basename "$YAML" .yaml)
non_yaml_part="${directory}/${filename}"

LOGDIR="log/$non_yaml_part"
mkdir -p ${LOGDIR}
echo "log file: $LOGDIR"

# for scale in "${scales[@]}"; do
    # bash scripts/stop_mps.sh $MPS_FILE
    # bash scripts/start_mps.sh $MPS_FILE
# WORKLOAD="${workload_file}_${scale}.json"
WORKLOAD=${workload_file}
export CUDA_MPS_PIPE_DIRECTORY=$MPS_FILE/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=$MPS_FILE/nvidia-log

export PATH=/home/lurunyu/envs/miniconda3/envs/muxserve/bin/:$PATH
FLEXSM_SHM_PREFIX="placement_${filename}" python -m muxserve.launch ${YAML} \
    --nnodes=$NNODES --node-rank=0 --master-addr=127.0.0.1 \
    --nproc_per_node=$NPROCS \
    --schedule-approach adbs \
    --workload-file ${WORKLOAD} \
    --max-num-batched-tokens 2048 \
    --server-port $(get_available_port) --flexstore-port $(get_available_port) \
    2>&1 | tee ${LOGDIR}/log.log
# done
