#!/bin/bash
# the following must be performed with root privilege
# >>> sudo sh scripts/start_mps.sh [MPS_DIR]

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <mps_dir>"
    echo "bash scripts/start_mps.sh /mnt/afs/jfduan/LLMInfer/MuxServe/log/mps"
    exit 1
fi

MPSDIR=$1

mkdir -p $MPSDIR

# export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=${MPSDIR}/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=${MPSDIR}/nvidia-log

nvidia-cuda-mps-control -d

mkdir -p ${MPSDIR}/nvidia-mps
mkdir -p ${MPSDIR}/nvidia-log

# change the permission of the pipe directory
chmod 777 ${MPSDIR}/nvidia-log
