export PYTHONPATH=$(pwd):$PYTHONPATH

logdir="log/bench_7b_13b_30b_same"

# Temporal multiplexing
mkdir -p ${logdir}/temporal
for rate in 13 11 10 1; do
    echo "Temporal multiplexing with sharegpt_n3_rate${rate}.json"
    python -m muxserve.launch examples/workloads/cfg_temporal_n3.yaml \
        --workload-file examples/workloads/sharegpt_n3_rate${rate}.json \
        2>&1 | tee ${logdir}/temporal/temporal_7b_13b_30b_bs256_rate${rate}.log
    echo "\n\n"
    kill -9 $(pgrep -f muxserve)
    sleep 3
done
