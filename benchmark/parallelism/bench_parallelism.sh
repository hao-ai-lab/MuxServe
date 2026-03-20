#!/bin/bash
dir=$(dirname $0)
workdir=$(realpath $dir/../..)

export PYTHONPATH=$(pwd):$PYTHONPATH

LOGDIR="log/bench_parallelism"
# LOGDIR="log"
mkdir -p ${LOGDIR}

IFS=',' read -ra models <<< "$1"
IFS=',' read -ra ngpus <<< "$2"

MPSDIR="${workdir}/log/mps"

for model in ${models[@]}; do
    for ngpu in ${ngpus[@]}; do
        cfg="llama-${model}_n${ngpu}.yaml"
        if [ ! -f "benchmark/parallelism/${cfg}" ]; then
            echo "Config file ${cfg} does not exist!"
            continue
        fi
        echo "Running ${cfg}..."
        echo "djf@123" | sudo -S sh scripts/start_mps.sh ${MPSDIR}
        flexstore_port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
        python benchmark/parallelism/benchmark_parallelism.py \
            benchmark/parallelism/${cfg} \
            --nproc-per-node ${ngpu} \
            --mps-dir ${MPSDIR} \
            --workload-file examples/workloads/sharegpt_n1_rate10.json \
            --server-port 4134 --flexstore-port ${flexstore_port} \
            --log-dir ${LOGDIR} \
            2>&1 | tee log/muxserve_test.log
        echo "djf@123" | sudo -S sh scripts/stop_mps.sh ${MPSDIR}
    done
done
