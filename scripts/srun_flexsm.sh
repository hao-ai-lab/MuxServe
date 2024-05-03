#!/bin/bash
dir=$(dirname $0)
workdir=$(realpath $dir/..)

CUDA_PATH=/usr/local/cuda-11.8

NNODES=1

model_config=$1
workload_file=$2
NPROC_PER_NODE=$3
muxserve_log=$4
sub_mps_dir=$5
proc_id=$6
cluster=$7

mps_log_dir=log/mps/$sub_mps_dir

echo "============================================="
echo "model_config:     $model_config"
echo "workload_file:    $workload_file"
echo "NPROC_PER_NODE:   $NPROC_PER_NODE"
echo "muxserve_log:       $muxserve_log"
echo "mps_log_dir:      $mps_log_dir"
echo "proc_id:          $proc_id"
echo "cluster:          $cluster"
echo "vllm_log_dir:     $VLLM_PROC_LOG"
echo "============================================="
mkdir -p $mps_log_dir
mkdir -p $(dirname $muxserve_log)
mkdir -p $VLLM_PROC_LOG

usrname=dmhj
resource=N2lS.Ie.I60.${NPROC_PER_NODE}
if [ ${NNODES} -le 1 ]; then
    DIST=StandAlone
    DIST_ARG="--nproc_per_node=$NPROC_PER_NODE"
else
    DIST=AllReduce
    DIST_ARG="--nnodes=\$WORLD_SIZE --nproc_per_node=$NPROC_PER_NODE \
              --master-addr=\$MASTER_ADDR --master-port=\$MASTER_PORT \
              --node-rank=\$RANK"
fi
name=test
Image_ID=registry.cn-sh-01.sensecore.cn/cpii-ccr/clouddev-snapshot:20240102-14h01m22s

# cluster1 afb99c73-b2be-428d-963c-352460ab84cd d43c2524-492e-4df1-be8b-a95e688bd0f7
# cluster2 fa7dc572-ab64-4ad1-b7f2-335823fc8781 d43c2524-492e-4df1-be8b-a95e688bd0f7

# if 8 gpus => cluster1; 4 gpus => cluster2
if [ "$cluster" -eq 1 ]; then
  partition_id=afb99c73-b2be-428d-963c-352460ab84cd
  workspace_id=d43c2524-492e-4df1-be8b-a95e688bd0f7
elif [ "$cluster" -eq 2 ]; then
  partition_id=fa7dc572-ab64-4ad1-b7f2-335823fc8781
  workspace_id=d43c2524-492e-4df1-be8b-a95e688bd0f7
else
  echo "Invalid cluster value: $cluster"
  exit 1
fi


# --begin "2024-01-27T03:30" \
srun -p $partition_id \
    --workspace-id $workspace_id \
    --async -o log/srun \
    -N ${NNODES} -r ${resource} -j ${name} --framework pytorch -d ${DIST} \
    --container-image ${Image_ID}  \
    --container-mounts=e70f5aef-dd05-11ed-9103-ba18b4912d57:/mnt/afs \
    bash -c "rm -rf /usr/local/nvidia/lib64/*; \
    su - ${usrname} -c \" \
        cd ${workdir} &&  \
        export CUDA_PATH=${CUDA_PATH} && \
        export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${CUDA_PATH}/extras/CUPTI/lib64:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH && \
        export PATH=/mnt/afs/dmhj/miniconda3/envs/muxserve/bin:/mnt/afs/dmhj/miniconda3/condabin:$PATH && \
        export PATH=${CUDA_PATH}/bin:$PATH && \
        export PYTHONPATH=${workspace}:${PYTHONPATH} && \
        export CUDA_MPS_PIPE_DIRECTORY=$mps_log_dir/nvidia-mps && \
        export CUDA_MPS_LOG_DIRECTORY=$mps_log_dir/nvidia-log && \
        export VLLM_PROC_LOG=$VLLM_PROC_LOG && \
        echo \"dmhj@123\" | sudo -S sh $workdir/scripts/start_mps.sh $mps_log_dir && \
        python -m muxserve.launch $model_config \
                    --nproc_per_node=$NPROC_PER_NODE \
                    --mps-dir $mps_log_dir \
                    --workload-file $workload_file \
                    --max-num-batched-tokens 2048 \
                    --server-port 48${proc_id}2 --flexstore-port 58${proc_id}1 \
                    --schedule-approach adbs \
                    ${DIST_ARG} \
                  2>&1 | tee $muxserve_log && \
        echo -e \"\n\n\" && \
        echo \"dmhj@123\" | sudo -S sh $workdir/scripts/stop_mps.sh $mps_log_dir && \
        kill -9 \$(pgrep -f 58${proc_id}1) && \
        sleep 180 \""
