dir=$(dirname $0)
workdir=$(realpath $dir/..)

launch_type=$1
model_config_prefix=$2
workloads=$(realpath $3)

if [ $# -ne 3 ]; then
    echo "Usage: $0 <launch_type> <model_config list> <workloads_file>"
    echo "bash scripts/bench_end_to_end.sh muxserve   ./benchmark/model_cfgs/models_GPUnum16_mesh_size\*  ./benchmark/workloads/sharegpt.json"
    echo "bash scripts/bench_end_to_end.sh temporal ./benchmark/model_cfgs/models_GPUnum16_mesh_size\*  ./benchmark/workloads/sharegpt.json"
    echo "bash scripts/bench_end_to_end.sh spatial  ./benchmark/model_cfgs/spatial_models_GPUnum16.yaml ./benchmark/workloads/sharegpt.json"
    exit 1
fi

# specify `start_script` according to `launch_type`
if [ $launch_type = "spatial" ]; then
    start_script=$(realpath "./scripts/srun_spatial.sh")
elif [ $launch_type = "muxserve" ]; then
    start_script=$(realpath "./scripts/srun_muxserve.sh")
elif [ $launch_type = "temporal" ]; then
    start_script=$(realpath "./scripts/srun_temporal.sh")
else
    echo "Launch type $launch_type invalid"
    exit 1
fi

model_configs=$(find $model_config_prefix | tr "\n" ",")

# store all the model_config in `configs`
IFS="," read -ra configs <<< "$model_configs"
echo "configs: $configs"

end_to_end_log_dir="$workdir/log/end_to_end"
mkdir -p $end_to_end_log_dir

# workload_rate=$(echo "$workloads" | sed 's/.*_rate\([0-9]\+\).*/\1/')
proc_id=25


ts=$(date +"%m%d%H%M")
export VLLM_PROC_LOG="$end_to_end_log_dir/vllm_proc/$ts"

llm_id=0
if [ $launch_type = "spatial" ]; then
    spatial_config=$(realpath $configs)

    # 读取文件的第一行
    first_line=$(head -n 1 "$spatial_config")
    # 去除行首的 "# " 部分
    ngpu_per_llm=${first_line#*# }
    # 分割数字并迭代每个数字
    IFS='_' read -ra ngpu_arr <<< "$ngpu_per_llm"

    alpha=$(echo "$spatial_config" | sed -n 's/.*alpha\([0-9.]*\).*/\1/p')
    scale=$(echo "$spatial_config" | sed -n 's/.*scale\([0-9.]*\).*/\1/p')

    cluster=2
    # if [ "$mesh_size" -eq 8 ]; then
    #     cluster=1
    # else
    #     cluster=2
    # fi
    for ngpus in "${ngpu_arr[@]}"; do
        spatial_log="$end_to_end_log_dir/spatial/alpha${alpha}_${scale}_${ts}/llm_$llm_id.log"

        echo -e "\033[31mcommand:\033[0m\n\t\033[35mbash\033[0m $start_script $spatial_config $workloads $llm_id $ngpus $spatial_log $proc_id $cluster"
        VLLM_PROC_LOG=$VLLM_PROC_LOG bash $start_script $spatial_config $workloads $llm_id $ngpus $spatial_log $proc_id $cluster
        (( llm_id++ ))
    done

elif [ $launch_type = "muxserve" ]; then
    for config in "${configs[@]}"; do
        model_config=$(realpath $config)
        mesh_size=$(echo "$model_config" | sed 's/.*mesh_size\([0-9]\+\).*/\1/')
        idx=$(echo "$model_config" | sed 's/.*idx\([0-9]\+\).*/\1/')
        alpha=$(echo "$model_config" | sed -n 's/.*alpha\([0-9.]*\).*/\1/p')
        scale=$(echo "$model_config" | sed -n 's/.*scale\([0-9.]*\).*/\1/p')

        ngpus=$mesh_size

        mps_dir="${ts}_mesh${mesh_size}_idx${idx}"
        muxserve_log="${end_to_end_log_dir}/muxserve/alpha${alpha}_${scale}_${ts}/mesh${mesh_size}idx${idx}.log"

        cluster=2
        # if [ "$mesh_size" -eq 8 ]; then
        #     cluster=1
        # else
        #     cluster=2
        # fi

        echo -e "\033[31mcommand:\033[0m\n\t\033[35mbash\033[0m $start_script $model_config $workloads $ngpus $muxserve_log $mps_dir $proc_id $cluster"
        VLLM_PROC_LOG=$VLLM_PROC_LOG bash $start_script $model_config $workloads $ngpus $muxserve_log $mps_dir $proc_id $cluster
        (( proc_id++ ))
    done
elif [ $launch_type = "temporal" ]; then
    for config in "${configs[@]}"; do
        model_config=$(realpath $config)
        mesh_size=$(echo "$model_config" | sed 's/.*mesh_size\([0-9]\+\).*/\1/')
        idx=$(echo "$model_config" | sed 's/.*idx\([0-9]\+\).*/\1/')
        alpha=$(echo "$model_config" | sed -n 's/.*alpha\([0-9.]*\).*/\1/p')
        scale=$(echo "$model_config" | sed -n 's/.*scale\([0-9.]*\).*/\1/p')

        ngpus=$mesh_size

        temporal_log="${end_to_end_log_dir}/temporal/alpha${alpha}_${scale}_${ts}/mesh${mesh_size}idx${idx}.log"

        cluster=2

        echo -e "\033[31mcommand:\033[0m\n\t\033[35mbash\033[0m $start_script $model_config $workloads $ngpus $temporal_log $proc_id $cluster"
        VLLM_PROC_LOG=$VLLM_PROC_LOG bash $start_script $model_config $workloads $ngpus $temporal_log $proc_id $cluster
        (( proc_id++ ))
    done
else
    echo "Launch type $launch_type invalid"
    exit 1
fi
