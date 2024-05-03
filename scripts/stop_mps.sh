#!/bin/bash
# the following must be performed with root privilege
# >>> sudo sh scripts/stop_mps.sh

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <mps_dir>"
    echo "bash scripts/stop_mps.sh /mnt/afs/jfduan/LLMInfer/MuxServe/log/mps"
    exit 1
fi

MPSDIR=$1

echo quit | nvidia-cuda-mps-control
pkill -f nvidia-cuda-mps-control

rm -rf ${MPSDIR}/nvidia-mps
rm -rf ${MPSDIR}/nvidia-log
